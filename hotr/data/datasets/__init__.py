# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from hotr.data.datasets.coco import build as build_coco
from hotr.data.datasets.vcoco import build as build_vcoco
from hotr.data.datasets.hico import build as build_hico

def get_coco_api_from_dataset(dataset):
    for _ in range(10): # what is this for?
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    elif args.dataset_file == 'hico-det':
        return build_hico(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')