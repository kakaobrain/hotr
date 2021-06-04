# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime

from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP

class HOTR(nn.Module):
    def __init__(self, detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss):
        super().__init__()

        # * Instance Transformer ---------------
        self.detr = detr
        if freeze_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_hoi_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions+1)
        # --------------------------------------------------------------------

        # * Transformer Options ---------------------------------------------
        self.interaction_transformer = interaction_transformer

        if share_enc: # share encoder
            self.interaction_transformer.encoder = detr.transformer.encoder

        if pretrained_dec: # free variables for interaction decoder
            self.interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.interaction_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------

        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss
        # ----------------------------------

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        start_time = time.time()
        hs, _ = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1])
        inst_repr = F.normalize(hs[-1], p=2, dim=2) # instance representations

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        object_detection_time = time.time() - start_time
        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        start_time = time.time()
        assert hasattr(self, 'interaction_transformer'), "Missing Interaction Transformer."
        interaction_hs = self.interaction_transformer(self.detr.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] # interaction representations
    
        # [HO Pointers]
        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)
        outputs_hidx = [(torch.bmm(H_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]
        
        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        # --------------------------------------------------
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)
        # -------------------------------------------------------------------

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }

        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1])]