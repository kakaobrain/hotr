# Copyright (c) Kakaobrain, Inc. and its affiliates. All Rights Reserved
"""
V-COCO dataset which returns image_id for evaluation.
"""
from pathlib import Path

from PIL import Image
import os
import numpy as np
import json
import torch
import torch.utils.data
import torchvision

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from hotr.data.datasets import builtin_meta
import hotr.data.transforms.transforms as T

class VCocoDetection(Dataset):
    def __init__(self,
                 img_folder,
                 ann_file,
                 all_file,
                 filter_empty_gt=True,
                 transforms=None):
        self.img_folder = img_folder
        self.file_meta = dict()
        self._transforms = transforms

        self.ann_file = ann_file
        self.all_file = all_file
        self.filter_empty_gt = filter_empty_gt

        # COCO initialize
        self.coco = COCO(self.all_file)
        self.COCO_CLASSES = builtin_meta._get_coco_instances_meta()['coco_classes']
        self.file_meta['coco_classes'] = self.COCO_CLASSES

        # Load V-COCO Dataset
        self.vcoco_all = self.load_vcoco(self.ann_file)

        # Save COCO annotation data
        self.image_ids = sorted(list(set(self.vcoco_all[0]['image_id'].reshape(-1))))

        # Filter Data
        if filter_empty_gt:
            self.filter_image_id()
        self.img_infos = self.load_annotations()

        # Refine Data
        self.save_action_name()
        self.mapping_inst_action_to_action()
        self.load_subobj_classes()
        self.CLASSES = self.act_list

    ############################################################################
    # Load V-COCO Dataset
    ############################################################################
    def load_vcoco(self, dir_name=None):
        with open(dir_name, 'rt') as f:
            vsrl_data = json.load(f)

        for i in range(len(vsrl_data)):
            vsrl_data[i]['role_object_id'] = np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']),-1)).T
            for j in ['ann_id', 'label', 'image_id']:
                vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1,1))

        return vsrl_data

    ############################################################################
    # Refine Data
    ############################################################################
    def save_action_name(self):
        self.inst_act_list = list()
        self.act_list = list()

        # add instance action human classes
        self.num_subject_act = 0
        for vcoco in self.vcoco_all:
            self.inst_act_list.append('human_' + vcoco['action_name'])
            self.num_subject_act += 1

        # add instance action object classes
        for vcoco in self.vcoco_all:
            if len(vcoco['role_name']) == 3:
                self.inst_act_list.append('object_' + vcoco['action_name']+'_'+vcoco['role_name'][1])
                self.inst_act_list.append('object_' + vcoco['action_name']+'_'+vcoco['role_name'][2])
            elif len(vcoco['role_name']) < 2:
                continue
            else:
                self.inst_act_list.append('object_' + vcoco['action_name']+'_'+vcoco['role_name'][-1]) # when only two roles

        # add action classes
        for vcoco in self.vcoco_all:
            if len(vcoco['role_name']) == 3:
                self.act_list.append(vcoco['action_name']+'_'+vcoco['role_name'][1])
                self.act_list.append(vcoco['action_name']+'_'+vcoco['role_name'][2])
            else:
                self.act_list.append(vcoco['action_name']+'_'+vcoco['role_name'][-1])

        # add to meta
        self.file_meta['action_classes'] = self.act_list

    def mapping_inst_action_to_action(self):
        sub_idx = 0
        obj_idx = self.num_subject_act

        self.sub_label_to_action = list()
        self.obj_label_to_action = list()

        for vcoco in self.vcoco_all:
            role_name = vcoco['role_name']

            self.sub_label_to_action.append(sub_idx)
            if len(role_name) == 3 :
                self.sub_label_to_action.append(sub_idx)
                self.obj_label_to_action.append(obj_idx)
                self.obj_label_to_action.append(obj_idx+1)
                obj_idx += 2
            elif len(role_name) == 2:
                self.obj_label_to_action.append(obj_idx)
                obj_idx += 1
            else:
                self.obj_label_to_action.append(0)

            sub_idx += 1

    def load_subobj_classes(self):
        self.vcoco_labels = dict()
        for img in self.image_ids:
            self.vcoco_labels[img] = dict()
            self.vcoco_labels[img]['boxes'] = np.empty((0, 4), dtype=np.float32)
            self.vcoco_labels[img]['categories'] = np.empty((0), dtype=np.int32)

            ann_ids = self.coco.getAnnIds(imgIds=img, iscrowd=None)
            objs = self.coco.loadAnns(ann_ids)

            valid_ann_ids = []

            for i, obj in enumerate(objs):
                if 'ignore' in obj and obj['ignore'] == 1: continue

                x1 = obj['bbox'][0]
                y1 = obj['bbox'][1]
                x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
                y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)

                if obj['area'] > 0 and x2 > x1 and y2 > y1:
                    bbox = np.array([x1, y1, x2, y2]).reshape(1, -1)
                    cls = obj['category_id']
                    self.vcoco_labels[img]['boxes'] = np.concatenate([self.vcoco_labels[img]['boxes'], bbox], axis=0)
                    self.vcoco_labels[img]['categories'] = np.concatenate([self.vcoco_labels[img]['categories'], [cls]], axis=0)

                    valid_ann_ids.append(ann_ids[i])

            num_valid_objs = len(valid_ann_ids)

            self.vcoco_labels[img]['agent_actions'] = -np.ones((num_valid_objs, self.num_action()), dtype=np.int32)
            self.vcoco_labels[img]['obj_actions'] = np.zeros((num_valid_objs, self.num_action()), dtype=np.int32)
            self.vcoco_labels[img]['role_id'] = -np.ones((num_valid_objs, self.num_action()), dtype=np.int32)

            for ix, ann_id in enumerate(valid_ann_ids):
                in_vcoco = np.where(self.vcoco_all[0]['ann_id'] == ann_id)[0]
                if in_vcoco.size > 0:
                    self.vcoco_labels[img]['agent_actions'][ix, :] = 0

                    agent_act_id = 0
                    obj_act_id = -1
                    for i, x in enumerate(self.vcoco_all):
                        has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
                        if has_label.size > 0:
                            assert has_label.size == 1
                            rids = x['role_object_id'][has_label]

                            if rids.shape[1] == 3:
                                self.vcoco_labels[img]['agent_actions'][ix, agent_act_id] = 1
                                self.vcoco_labels[img]['agent_actions'][ix, agent_act_id+1] = 1
                                agent_act_id += 2
                            else:
                                self.vcoco_labels[img]['agent_actions'][ix, agent_act_id] = 1
                                agent_act_id += 1
                                if rids.shape[1] == 1 : obj_act_id += 1

                            for j in range(1, rids.shape[1]):
                                obj_act_id += 1
                                if rids[0, j] == 0: continue # no role
                                aid = np.where(valid_ann_ids == rids[0, j])[0]

                                self.vcoco_labels[img]['role_id'][ix, obj_act_id] = aid
                                self.vcoco_labels[img]['obj_actions'][aid, obj_act_id] = 1

                        else:
                            rids = x['role_object_id'][0]
                            if rids.shape[0] == 3:
                                agent_act_id += 2
                                obj_act_id += 2
                            else:
                                agent_act_id += 1
                                obj_act_id += 1

    ############################################################################
    # Annotation Loader
    ############################################################################
    # >>> 1. instance
    def load_instance_annotations(self, image_index):
        num_ann = self.vcoco_labels[image_index]['boxes'].shape[0]
        inst_action = np.zeros((num_ann, self.num_inst_action()), np.int)
        inst_bbox = np.zeros((num_ann, 4), dtype=np.float32)
        inst_category = np.zeros((num_ann, ), dtype=np.int)

        for idx in range(num_ann):
            inst_bbox[idx] = self.vcoco_labels[image_index]['boxes'][idx]
            inst_category[idx]= self.vcoco_labels[image_index]['categories'][idx] #+ 1 # category 1 ~ 81

            if inst_category[idx] == 1:
                act = self.vcoco_labels[image_index]['agent_actions'][idx]
                inst_action[idx, :self.num_subject_act] = act[np.unique(self.sub_label_to_action, return_index=True)[1]]

                # when person is the obj
                act = self.vcoco_labels[image_index]['obj_actions'][idx] # when person is the obj
                if act.any():
                    inst_action[idx, self.num_subject_act:] = act[np.nonzero(self.obj_label_to_action)[0]]
                    if inst_action[idx, :self.num_subject_act].sum(axis=-1) < 0:
                        inst_action[idx, :self.num_subject_act] = 0
            else:
                act = self.vcoco_labels[image_index]['obj_actions'][idx]
                inst_action[idx, self.num_subject_act:] = act[np.nonzero(self.obj_label_to_action)[0]]

        # >>> For Objects that are in COCO but not in V-COCO,
        # >>> Human -> [-1 * 26, 0 * 25]
        # >>> Object -> [0 * 51]
        # >>> Don't return anything for actions with max 0 or max -1
        max_val = inst_action.max(axis=1)
        if (max_val > 0).sum() == 0:
            print(f"No Annotations for {image_index}")
            print(inst_action)
            print(self.vcoco_labels[image_index]['agent_actions'][idx])
            print(self.vcoco_labels[image_index]['obj_actions'][idx])

        return inst_bbox[max_val > 0], inst_category[max_val > 0], inst_action[max_val > 0]

    # >>> 2. pair
    def load_pair_annotations(self, image_index):
        num_ann = self.vcoco_labels[image_index]['boxes'].shape[0]
        pair_action = np.zeros((0, self.num_action()), np.int)
        pair_bbox = np.zeros((0, 8), dtype=np.float32)
        pair_target = np.zeros((0, ), dtype=np.int)

        for idx in range(num_ann):
            h_box = self.vcoco_labels[image_index]['boxes'][idx]
            h_cat = self.vcoco_labels[image_index]['categories'][idx]
            if h_cat != 1 : continue # human_id = 1

            h_act = self.vcoco_labels[image_index]['agent_actions'][idx]
            if np.any((h_act==-1)) : continue

            o_act = dict()
            for aid in range(self.num_action()):
                if h_act[aid] == 0 : continue
                o_id = self.vcoco_labels[image_index]['role_id'][idx, aid]
                if o_id not in o_act : o_act[o_id] = list()
                o_act[o_id].append(aid)

            for o_id in o_act.keys():
                if o_id == -1:
                    o_box = -np.ones((4, ))
                    o_cat = -1 # target is background
                else:
                    o_box = self.vcoco_labels[image_index]['boxes'][o_id]
                    o_cat = self.vcoco_labels[image_index]['categories'][o_id] # category 0 ~ 80

                box = np.concatenate([h_box, o_box]).astype(np.float32)
                act = np.zeros((1, self.num_action()), np.int)
                tar = np.zeros((1, ), np.int)
                tar[0] = o_cat #+ 1 # category 1 ~ 81
                for o_aid in o_act[o_id] : act[0, o_aid] = 1

                pair_action = np.concatenate([pair_action, act], axis=0)
                pair_bbox = np.concatenate([pair_bbox, np.expand_dims(box, axis=0)], axis=0)
                pair_target = np.concatenate([pair_target, tar], axis=0)

        return pair_bbox, pair_action, pair_target

    # >>> 3. image infos
    def load_annotations(self):
        img_infos = []
        for i in self.image_ids:
            info = self.coco.loadImgs([i])[0]
            img_infos.append(info)
        return img_infos

    ############################################################################
    # Check Method
    ############################################################################
    def sum_action_ann_for_id(self, find_idx):
        sum = 0
        for action_ann in self.vcoco_all:
            img_ids = action_ann['image_id']
            img_labels = action_ann['label']

            final_inds = img_ids[img_labels == 1]

            if (find_idx in final_inds):
                sum += 1
        # sum of class-wise existence
        return (sum > 0)

    def filter_image_id(self):
        empty_gt_list = []
        for img_id in self.image_ids:
            if not self.sum_action_ann_for_id(img_id):
                empty_gt_list.append(img_id)

        for remove_id in empty_gt_list:
            rm_idx = self.image_ids.index(remove_id)
            self.image_ids.remove(remove_id)

    ############################################################################
    # Preprocessing
    ############################################################################
    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        image = Image.open(os.path.join(self.img_folder, img_info['file_name'])).convert('RGB')
        target = self.get_ann_info(idx)

        w, h = image.size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(image, target) # "size" gets converted here

        return img, target

    ############################################################################
    # Get Method
    ############################################################################
    def __getitem__(self, idx):
        img, target = self.prepare_img(idx)
        return img, target

    def __len__(self):
        return len(self.image_ids)

    def get_human_label_idx(self):
        return self.sub_label_to_action

    def get_object_label_idx(self):
        return self.obj_label_to_action

    def get_image_ids(self):
        return self.image_ids

    def get_categories(self):
        return self.COCO_CLASSES

    def get_inst_action(self):
        return self.inst_act_list

    def get_actions(self):
        return self.act_list

    def get_human_action(self):
        return self.inst_act_list[:self.num_subject_act]

    def get_object_action(self):
        return self.inst_act_list[self.num_subject_act:]

    def get_ann_info(self, idx):
        img_idx = int(self.image_ids[idx])

        # load each annotation
        inst_bbox, inst_label, inst_actions = self.load_instance_annotations(img_idx)
        pair_bbox, pair_actions, pair_targets = self.load_pair_annotations(img_idx)

        sample = {
            'image_id' : torch.tensor([img_idx]),
            'boxes': torch.as_tensor(inst_bbox, dtype=torch.float32),
            'labels': torch.tensor(inst_label, dtype=torch.int64),
            'inst_actions': torch.tensor(inst_actions, dtype=torch.int64),
            'pair_boxes': torch.as_tensor(pair_bbox, dtype=torch.float32),
            'pair_actions': torch.tensor(pair_actions, dtype=torch.int64),
            'pair_targets': torch.tensor(pair_targets, dtype=torch.int64),
        }

        return sample

    ############################################################################
    # Number Method
    ############################################################################
    def num_category(self):
        return len(self.COCO_CLASSES)

    def num_action(self):
        return len(self.act_list)

    def num_inst_action(self):
        return len(self.inst_act_list)

    def num_human_act(self):
        return len(self.inst_act_list[:self.num_subject_act])

    def num_object_act(self):
        return len(self.inst_act_list[self.num_subject_act:])

def make_hoi_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided V-COCO path {root} does not exist'
    PATHS = {
        "train": (root / "coco/images/train2014/", root / "data/vcoco" / 'vcoco_trainval.json'),
        "val": (root / "coco/images/val2014", root / "data/vcoco" / 'vcoco_test.json'),
        "test": (root / "coco/images/val2014", root / "data/vcoco" / 'vcoco_test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    all_file = root / "data/instances_vcoco_all_2014.json"
    dataset = VCocoDetection(
        img_folder = img_folder,
        ann_file = ann_file,
        all_file = all_file,
        filter_empty_gt=True,
        transforms = make_hoi_transforms(image_set)
    )
    dataset.file_meta['dataset_file'] = args.dataset_file
    dataset.file_meta['image_set'] = image_set

    return dataset
