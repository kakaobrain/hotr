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
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class HOTR(nn.Module):
    def __init__(self, detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss,
                 use_pos_info=False,
                 return_obj_class=None):
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

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1]+1]
        # ------------------------------------------------------------------

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

        # *************************************
        # * if add the position info to query *
        # 如果要使用pos info，那么只要是对decoder中的模块进行操作
        self.use_pos_info=use_pos_info
        if self.use_pos_info:
            # 进行标志的设置
            self.interaction_transformer.decoder.use_pos_info=True
            # 获取三个预测头
            self.interaction_transformer.decoder.H_Pointer_embed=self.H_Pointer_embed
            self.interaction_transformer.decoder.O_Pointer_embed=self.O_Pointer_embed
            self.interaction_transformer.decoder.bbox_embed=self.detr.bbox_embed

            # repr scaler
            # self.repr_scaler=None

            # 暂时假设每个层的MLP都是不同的吧
            # scaler的输出维度可以是d model也可以是1，这个可以进行实验
            # scaler可以每一层共享参数，也可以每一层都一样
            d_model=hidden_dim
            self.interaction_transformer.decoder.H_Pointer_scaler = MLP(d_model, d_model, d_model, 2)
            self.interaction_transformer.decoder.O_Pointer_scaler = MLP(d_model, d_model, d_model, 2)
            self.interaction_transformer.decoder.Pointer_proj = MLP(2 * d_model, d_model, d_model, 3)
            self.interaction_transformer.decoder.pos_scaler = MLP(d_model, d_model, d_model, 2)
        # **************************************

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

        # *********************************************
        # 如果是使用了pos info，那么输出的时候连通指针进行输出
        if not self.use_pos_info:
            interaction_hs = self.interaction_transformer(self.detr.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] # interaction representations
            # [HO Pointers]
            H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
            O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)
        else:
            interaction_hs,memory,H_Pointer_reprs,O_Pointer_reprs=self.interaction_transformer(self.detr.input_proj(src),mask,self.query_embed.weight,pos[-1])
            H_Pointer_reprs=F.normalize(H_Pointer_reprs,p=2,dim=-1)
            O_Pointer_reprs=F.normalize(O_Pointer_reprs,p=2,dim=-1)
        outputs_hidx = [(torch.bmm(H_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, inst_repr.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]
        # *********************************************
        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        # --------------------------------------------------
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)
        # -------------------------------------------------------------------

        # [Target Classification]
        if self.return_obj_class:
            detr_logits = outputs_class[-1, ..., self._valid_obj_ids]
            o_indices = [output_oidx.max(-1)[-1] for output_oidx in outputs_oidx]
            obj_logit_stack = [torch.stack([detr_logits[batch_, o_idx, :] for batch_, o_idx in enumerate(o_indice)], 0) for o_indice in o_indices]
            outputs_obj_class = obj_logit_stack

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }

        if self.return_obj_class: out["pred_obj_logits"] = outputs_obj_class[-1]

        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = \
                self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_obj_class) \
                if self.return_obj_class else \
                self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

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

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_tgt):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_tgt[:-1])]