# ------------------------------------------------------------------------
# HOTR official code : main.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import hotr.data.datasets as datasets
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.data.datasets import build_dataset, get_coco_api_from_dataset
from hotr.engine.trainer import train_one_epoch
from hotr.engine import hoi_evaluator, hoi_accumulator
from hotr.models import build_model
import wandb

from hotr.util.logger import print_params, print_args

def save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename):
    # save_ckpt: function for saving checkpoints
    output_dir = Path(args.output_dir)
    if args.output_dir:
        checkpoint_path = output_dir / f'{filename}.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val' if not args.eval else 'test', args=args)
    assert dataset_train.num_action() == dataset_val.num_action(), "Number of actions should be the same between splits"
    args.num_classes = dataset_train.num_category()
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == 'vcoco':
        # Save V-COCO dataset statistics
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
    elif args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
    print_args(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Weight Setup
    if args.frozen_weights is not None:
        if args.frozen_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.frozen_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        # test only mode
        if args.HOIDet:
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                sc1, sc2 = hoi_accumulator(args, total_res, True, False)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f}')
                print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f}')
                print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f}')
            else: raise ValueError(f'dataset {args.dataset_file} is not supported.')
            return
        else:
            test_stats, coco_evaluator = evaluate_coco(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return

    # stats
    scenario1, scenario2 = 0, 0
    best_mAP, best_rare, best_non_rare = 0, 0, 0

    # add argparse
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)

    # Training starts here!
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs,
            args.clip_max_norm, dataset_file=args.dataset_file, log=args.wandb)
        lr_scheduler.step()

        # Validation
        if args.validate:
            print('-'*100)
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    sc1, sc2 = hoi_accumulator(args, total_res, False, args.wandb)
                    if sc1 > scenario1:
                        scenario1 = sc1
                        scenario2 = sc2
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| Scenario #1 mAP : {sc1:.2f} ({scenario1:.2f})')
                    print(f'| Scenario #2 mAP : {sc2:.2f} ({scenario2:.2f})')
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    if test_stats['mAP'] > best_mAP:
                        best_mAP = test_stats['mAP']
                        best_rare = test_stats['mAP rare']
                        best_non_rare = test_stats['mAP non-rare']
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f} ({best_mAP:.2f})')
                    print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f} ({best_rare:.2f})')
                    print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f} ({best_non_rare:.2f})')
                    if args.wandb and utils.get_rank() == 0:
                        wandb.log({
                            'mAP': test_stats['mAP']
                        })
            print('-'*100)
        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='checkpoint')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.dataset_file == 'vcoco':
        print(f'| Scenario #1 mAP : {scenario1:.2f}')
        print(f'| Scenario #2 mAP : {scenario2:.2f}')
    elif args.dataset_file == 'hico-det':
        print(f'| mAP (full)\t\t: {best_mAP:.2f}')
        print(f'| mAP (rare)\t\t: {best_rare:.2f}')
        print(f'| mAP (non-rare)\t: {best_non_rare:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
