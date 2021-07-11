# ------------------------------------------------------------------------
# HOTR official code : main.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import copy
import numpy as np

from torch import nn

from hotr.util import box_ops
from hotr.util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_actions=None, HOI_losses=None, HOI_matcher=None, args=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef=eos_coef

        self.HOI_losses = HOI_losses
        self.HOI_matcher = HOI_matcher

        if args:
            self.HOI_eos_coef = args.hoi_eos_coef
            if args.dataset_file == 'vcoco':
                self.invalid_ids = args.invalid_ids
                self.valid_ids = np.concatenate((args.valid_ids,[-1]), axis=0) # no interaction
            elif args.dataset_file == 'hico-det':
                self.invalid_ids = []
                self.valid_ids = list(range(num_actions)) + [-1]

                # for targets
                self.num_tgt_classes = len(args.valid_obj_ids)
                tgt_empty_weight = torch.ones(self.num_tgt_classes + 1)
                tgt_empty_weight[-1] = self.HOI_eos_coef
                self.register_buffer('tgt_empty_weight', tgt_empty_weight)
        self.dataset_file = args.dataset_file
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    #######################################################################################################################
    # * DETR Losses
    #######################################################################################################################
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    #######################################################################################################################
    # * HOTR Losses
    #######################################################################################################################
    # >>> HOI Losses 1 : HO Pointer
    def loss_pair_labels(self, outputs, targets, hoi_indices, num_boxes, log=False):
        assert ('pred_hidx' in outputs and 'pred_oidx' in outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        src_hidx = outputs['pred_hidx']
        src_oidx = outputs['pred_oidx']

        idx = self._get_src_permutation_idx(hoi_indices)

        target_hidx_classes = torch.full(src_hidx.shape[:2], -1, dtype=torch.int64, device=src_hidx.device)
        target_oidx_classes = torch.full(src_oidx.shape[:2], -1, dtype=torch.int64, device=src_oidx.device)

        # H Pointer loss        
        target_classes_h = torch.cat([t["h_labels"][J] for t, (_, J) in zip(targets, hoi_indices)])
        target_hidx_classes[idx] = target_classes_h

        # O Pointer loss
        target_classes_o = torch.cat([t["o_labels"][J] for t, (_, J) in zip(targets, hoi_indices)])
        target_oidx_classes[idx] = target_classes_o

        loss_h = F.cross_entropy(src_hidx.transpose(1, 2), target_hidx_classes, ignore_index=-1)
        loss_o = F.cross_entropy(src_oidx.transpose(1, 2), target_oidx_classes, ignore_index=-1)

        losses = {'loss_hidx': loss_h, 'loss_oidx': loss_o}

        return losses

    # >>> HOI Losses 2 : pair actions
    def loss_pair_actions(self, outputs, targets, hoi_indices, num_boxes):
        assert 'pred_actions' in outputs
        src_actions = outputs['pred_actions']
        idx = self._get_src_permutation_idx(hoi_indices)

        # Construct Target --------------------------------------------------------------------------------------------------------------
        target_classes_o = torch.cat([t["pair_actions"][J] for t, (_, J) in zip(targets, hoi_indices)])
        target_classes = torch.full(src_actions.shape, 0, dtype=torch.float32, device=src_actions.device)
        target_classes[..., -1] = 1 # the last index for no-interaction is '1' if a label exists

        pos_classes = torch.full(target_classes[idx].shape, 0, dtype=torch.float32, device=src_actions.device) # else, the last index for no-interaction is '0'
        pos_classes[:, :-1] = target_classes_o.float()
        target_classes[idx] = pos_classes
        # --------------------------------------------------------------------------------------------------------------------------------

        # BCE Loss -----------------------------------------------------------------------------------------------------------------------
        logits = src_actions.sigmoid()
        loss_bce = F.binary_cross_entropy(logits[..., self.valid_ids], target_classes[..., self.valid_ids], reduction='none')
        p_t = logits[..., self.valid_ids] * target_classes[..., self.valid_ids] + (1 - logits[..., self.valid_ids]) * (1 - target_classes[..., self.valid_ids])
        loss_bce = ((1-p_t)**2 * loss_bce)
        alpha_t = 0.25 * target_classes[..., self.valid_ids] + (1 - 0.25) * (1 - target_classes[..., self.valid_ids])
        loss_focal = alpha_t * loss_bce
        loss_act = loss_focal.sum() / max(target_classes[..., self.valid_ids[:-1]].sum(), 1)
        # --------------------------------------------------------------------------------------------------------------------------------

        losses = {'loss_act': loss_act}

        return losses

    # HOI Losses 3 : action targets
    def loss_pair_targets(self, outputs, targets, hoi_indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']
        idx = self._get_src_permutation_idx(hoi_indices)

        target_classes_o = torch.cat([t['pair_targets'][J] for t, (_, J) in zip(targets, hoi_indices)])
        pad_tgt = -1 # src_logits.shape[2]-1
        target_classes = torch.full(src_logits.shape[:2], pad_tgt, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.tgt_empty_weight, ignore_index=-1)
        losses = {'loss_tgt': loss_obj_ce}

        if log:
            ignore_idx = (target_classes_o != -1)
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx][ignore_idx, :-1], target_classes_o[ignore_idx])[0]
            # losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # *****************************************************************************
    # >>> DETR Losses
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # >>> HOTR Losses
    def get_HOI_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'pair_labels': self.loss_pair_labels,
            'pair_actions': self.loss_pair_actions
        }
        if self.dataset_file == 'hico-det': loss_map['pair_targets'] = self.loss_pair_targets
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    # *****************************************************************************

    def forward(self, outputs, targets, log=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if self.HOI_losses is not None:
            input_targets = [copy.deepcopy(target) for target in targets]
            hoi_indices, hoi_targets = self.HOI_matcher(outputs_without_aux, input_targets, indices, log)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # HOI detection losses
        if self.HOI_losses is not None:
            for loss in self.HOI_losses:
                losses.update(self.get_HOI_loss(loss, outputs, hoi_targets, hoi_indices, num_boxes))
            # if self.dataset_file == 'hico-det': losses['loss_oidx'] += losses['loss_tgt']

            if 'hoi_aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                    input_targets = [copy.deepcopy(target) for target in targets]
                    hoi_indices, targets_for_aux = self.HOI_matcher(aux_outputs, input_targets, indices, log)
                    for loss in self.HOI_losses:
                        kwargs = {}
                        if loss == 'pair_targets': kwargs = {'log': False} # Logging is enabled only for the last layer
                        l_dict = self.get_HOI_loss(loss, aux_outputs, hoi_targets, hoi_indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    # if self.dataset_file == 'hico-det': losses[f'loss_oidx_{i}'] += losses[f'loss_tgt_{i}']

        return losses