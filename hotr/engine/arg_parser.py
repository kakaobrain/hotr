# ------------------------------------------------------------------------
# HOTR official code : engine/arg_parser.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# Modified arguments are represented with *
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # DETR Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # DETR Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # DETR Transformer (= Encoder, Instance Decoder)
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss Option
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # Loss coefficients (DETR)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Matcher (DETR)
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * HOI Detection
    parser.add_argument('--HOIDet', action='store_true',
                        help="Train HOI Detection head if the flag is provided")
    parser.add_argument('--share_enc', action='store_true',
                        help="Share the Encoder in DETR for HOI Detection if the flag is provided")
    parser.add_argument('--pretrained_dec', action='store_true',
                        help="Use Pre-trained Decoder in DETR for Interaction Decoder if the flag is provided")                        
    parser.add_argument('--hoi_enc_layers', default=1, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_dec_layers', default=1, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_nheads', default=8, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--hoi_dim_feedforward', default=2048, type=int,
                        help="Number of decoding layers in HOI transformer")
    # parser.add_argument('--hoi_mode', type=str, default=None, help='[inst | pair | all]')
    parser.add_argument('--num_hoi_queries', default=100, type=int,
                        help="Number of Queries for Interaction Decoder")
    parser.add_argument('--hoi_aux_loss', action='store_true')


    # * HOTR Matcher
    parser.add_argument('--set_cost_idx', default=1, type=float,
                        help="IDX coefficient in the matching cost")
    parser.add_argument('--set_cost_act', default=1, type=float,
                        help="Action coefficient in the matching cost")
    parser.add_argument('--set_cost_tgt', default=1, type=float,
                        help="Target coefficient in the matching cost")

    # * HOTR Loss coefficients
    parser.add_argument('--temperature', default=0.05, type=float, help="temperature")
    parser.add_argument('--hoi_idx_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_act_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_tgt_loss_coef', default=1, type=float)
    parser.add_argument('--hoi_eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    # * dataset parameters
    parser.add_argument('--dataset_file', help='[coco | vcoco]')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--object_threshold', type=float, default=0, help='Threshold for object confidence')

    # machine parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--custom_path', default='',
                        help="Data path for custom inference. Only required for custom_main.py")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # mode
    parser.add_argument('--eval', action='store_true', help="Only evaluate results if the flag is provided")
    parser.add_argument('--validate', action='store_true', help="Validate after every epoch")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # * WanDB
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project_name', default='HOTR')
    parser.add_argument('--group_name', default='KakaoBrain')
    parser.add_argument('--run_name', default='run_000001')
    return parser
