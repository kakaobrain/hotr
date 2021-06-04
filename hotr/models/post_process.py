# ------------------------------------------------------------------------
# HOTR official code : hotr/models/post_process.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import time
import torch
import torch.nn.functional as F
from torch import nn
from hotr.util import box_ops

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, HOIDet):
        super().__init__()
        self.HOIDet = HOIDet

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, is_vcoco=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Preidction Branch for HOI detection
        if self.HOIDet:
            """ Compute HOI triplet prediction score.
            Our scoring function follows the implementation details of UnionDet.
            """
            out_time = outputs['hoi_recognition_time']

            start_time = time.time()
            pair_actions = torch.sigmoid(outputs['pred_actions'])
            h_prob = F.softmax(outputs['pred_hidx'], -1)
            h_idx_score, h_indices = h_prob.max(-1)

            o_prob = F.softmax(outputs['pred_oidx'], -1)
            o_idx_score, o_indices = o_prob.max(-1)
            hoi_recognition_time = (time.time() - start_time) + out_time

            results = []
            # iterate for batch size
            for batch_idx, (s, l, b) in enumerate(zip(scores, labels, boxes)):
                h_inds = (l == 1) & (s > threshold)
                o_inds = (s > threshold)

                h_box, h_cat = b[h_inds], s[h_inds]
                o_box, o_cat = b[o_inds], s[o_inds]

                if is_vcoco:
                    # for scenario 1 in v-coco dataset
                    o_inds = torch.cat((o_inds, torch.ones(1).type(torch.bool).to(o_inds.device)))
                    o_box = torch.cat((o_box, torch.Tensor([0, 0, 0, 0]).unsqueeze(0).to(o_box.device)))

                result_dict = {
                    'h_box': h_box, 'h_cat': h_cat,
                    'o_box': o_box, 'o_cat': o_cat,
                    'scores': s, 'labels': l, 'boxes': b
                }

                h_inds_lst = (h_inds == True).nonzero(as_tuple=False).squeeze(-1)
                o_inds_lst = (o_inds == True).nonzero(as_tuple=False).squeeze(-1)

                K = boxes.shape[1]
                n_act = pair_actions[batch_idx][:, :-1].shape[-1]
                score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
                sorted_score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
                id_score = torch.zeros((K, K+1)).to(pair_actions[batch_idx].device)

                # Score function
                for hs, h_idx, os, o_idx, pair_action in zip(h_idx_score[batch_idx], h_indices[batch_idx], o_idx_score[batch_idx], o_indices[batch_idx], pair_actions[batch_idx]):
                    matching_score = (1-pair_action[-1]) # no interaction score
                    if h_idx == o_idx and is_vcoco: o_idx = -1
                    if matching_score > id_score[h_idx, o_idx]:
                        id_score[h_idx, o_idx] = matching_score
                        sorted_score[:, h_idx, o_idx] = matching_score * pair_action[:-1]
                    score[:, h_idx, o_idx] += matching_score * pair_action[:-1]

                score += sorted_score
                score = score[:, h_inds, :]
                score = score[:, :, o_inds]

                result_dict.update({
                    'pair_score': score,
                    'hoi_recognition_time': hoi_recognition_time,
                })

                results.append(result_dict)
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
