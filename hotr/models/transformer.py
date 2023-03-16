# ------------------------------------------------------------------------
# HOTR official code : hotr/models/transformer.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR & HOTR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,d_model=d_model)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, return_decoder_input=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if not self.decoder.use_pos_info:
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            hs,H_Pointer_reprs,O_Pointer_reprs=self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
            # 这里return的memory没有用到，可能会出问题
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w),H_Pointer_reprs.transpose(1,2),O_Pointer_reprs.transpose(1,2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,d_model=512):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        # ******************************************
        # flag
        self.use_pos_info=False
        self.d_model=d_model
        # if self.use_pos_info:
        #     # pointer prediction head
        #     self.H_Pointer_embed = None
        #     self.O_Pointer_embed = None
        #     # repr scaler
        #     # self.repr_scaler=None
        #     #  bbox prediction head
        #     self.bbox_embed = None
        #     # 暂时假设每个层的MLP都是不同的吧
        #     # scaler的输出维度可以是d model也可以是1，这个可以进行实验
        #     # scaler可以每一层共享参数，也可以每一层都一样
        #     self.H_Pointer_scaler = None
        #     self.O_Pointer_scaler = None
        #     self.Pointer_proj = None
        #     self.pos_scaler = None
        # ******************************************
        # pointer prediction head
        self.H_Pointer_embed = None
        self.O_Pointer_embed = None
        # repr scaler
        # self.repr_scaler=None
        #  bbox prediction head
        self.bbox_embed = None
        # 暂时假设每个层的MLP都是不同的吧
        # scaler的输出维度可以是d model也可以是1，这个可以进行实验
        # scaler可以每一层共享参数，也可以每一层都一样
        self.H_Pointer_scaler = None
        self.O_Pointer_scaler = None
        self.Pointer_proj = None
        self.pos_scaler = None
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        # ***********************************
        if self.use_pos_info:
            output=query_pos
        o_intermediate=[]
        h_intermediate=[]
        # ***********************************
        # if use pos info, return the O pointers and the H pointers
        for layer_id,layer in enumerate(self.layers):
            if not self.use_pos_info:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos)
                if self.return_intermediate:
                    intermediate.append(self.norm(output))
            else:
                # output[query_num,bs,256]
                # 获取pointer repr
                H_Pointer_reprs=self.H_Pointer_embed(output)
                O_Pointer_reprs=self.O_Pointer_embed(output)
                # 获取pointer scaler
                if self.H_Pointer_scaler is not None and self.O_Pointer_scaler is not None:
                    H_Pointer_scaler=self.H_Pointer_scaler(output)
                    O_Pointer_scaler=self.O_Pointer_scaler(output)
                else:
                    H_Pointer_scaler=1
                    O_Pointer_scaler=1
                # 获取bbox
                # self.bbox_embed.eval()
                h_bbox=self.bbox_embed(H_Pointer_scaler*H_Pointer_reprs).sigmoid()
                o_bbox=self.bbox_embed(O_Pointer_scaler*O_Pointer_reprs).sigmoid()
                # self.bbox_embed.train()
                # 获取bbox的sin pos
                h_sin_embed=gen_sineembed_for_position(h_bbox)
                o_sin_embed=gen_sineembed_for_position(o_bbox)
                # 相加÷2
                query_sin_embed=(h_sin_embed+o_sin_embed)/2  # 2d
                # ***这里的query_pos1我之前没用上，这是一个bug
                query_pos1=self.Pointer_proj(query_sin_embed)
                # 获取第二个要相加的位置query
                if self.pos_scaler is not None:
                    query_pos2=query_sin_embed[...,:self.d_model]*self.pos_scaler(output)
                else:
                    query_pos2=query_sin_embed[...,:self.d_model]
                # 接下来是要修改transformer decoder layer的东西了，这里增加了一个参数query pos2进行拓展
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos1,query_pos2=query_pos2)
                # 确定一下是否返回里面的值
                if self.return_intermediate:
                    intermediate.append(self.norm(output))
                    if layer_id!=0:
                        h_intermediate.append(H_Pointer_reprs)
                        o_intermediate.append(O_Pointer_reprs)
        if self.use_pos_info:
            # 这里我又是改了一下bug
            H_Pointer_reprs=self.H_Pointer_embed(output)
            O_Pointer_reprs=self.O_Pointer_embed(output)
            h_intermediate.append(H_Pointer_reprs)
            o_intermediate.append(O_Pointer_reprs)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        # 额外增加了这个判断，主要是对输出结果进行了一定的规范输出
        if self.use_pos_info:
            if self.return_intermediate:
                return torch.stack(intermediate),torch.stack(h_intermediate),torch.stack(o_intermediate)
            else:
                # 如果上面的bug不改，这里返回的是上一层的，就会出大问题
                return output.unsqueeze(0),H_Pointer_reprs,O_Pointer_reprs
        else:
            if self.return_intermediate:
                return torch.stack(intermediate)

            return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_pos2:Optional[Tensor]=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ********************************
        # 如果query pos2是存在的，那么把query pos替换成query pos2
        if query_pos2 is not None:
            query_pos = query_pos2
        # ********************************
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    query_pos2:Optional[Tensor]=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # ********************************
        # 如果query pos2是存在的，那么把query pos替换成query pos2
        if query_pos2 is not None:
            query_pos=query_pos2
        # ********************************
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_pos2:Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,query_pos2)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,query_pos2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_hoi_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.hoi_nheads,
        dim_feedforward=args.hoi_dim_feedforward,
        num_encoder_layers=args.hoi_enc_layers,
        num_decoder_layers=args.hoi_dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")