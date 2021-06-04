# Copyright (c) KakaoBrain, Inc. and its affiliates. All Rights Reserved
"""
V-COCO evaluator that works in distributed mode.
"""
import os
import numpy as np
import torch

from hotr.util.misc import all_gather
from hotr.metrics.vcoco.ap_role import APRole
from functools import partial

def init_vcoco_evaluators(human_act_name, object_act_name):
    role_eval1 = APRole(act_name=object_act_name, scenario_flag=True, iou_threshold=0.5)
    role_eval2 = APRole(act_name=object_act_name, scenario_flag=False, iou_threshold=0.5)

    return role_eval1, role_eval2

class VCocoEvaluator(object):
    def __init__(self, args):
        self.img_ids = []
        self.eval_imgs = []
        self.role_eval1, self.role_eval2 = init_vcoco_evaluators(args.human_actions, args.object_actions)
        self.num_human_act = args.num_human_act
        self.action_idx = args.valid_ids

    def update(self, outputs):
        img_ids = list(np.unique(list(outputs.keys())))
        for img_num, img_id in enumerate(img_ids):
            print(f"Evaluating Score Matrix... : [{(img_num+1):>4}/{len(img_ids):<4}]" ,flush=True, end="\r")
            prediction = outputs[img_id]['prediction']
            target = outputs[img_id]['target']

            # score with prediction
            hbox, hcat, obox, ocat = list(map(lambda x: prediction[x], \
                ['h_box', 'h_cat', 'o_box', 'o_cat']))

            assert 'pair_score' in prediction
            score = prediction['pair_score']

            hbox, hcat, obox, ocat, score =\
                    list(map(lambda x: x.cpu().numpy(), [hbox, hcat, obox, ocat, score]))

            # ground-truth
            gt_h_inds = (target['labels'] == 1)
            gt_h_box = target['boxes'][gt_h_inds, :4].cpu().numpy()
            gt_h_act = target['inst_actions'][gt_h_inds, :self.num_human_act].cpu().numpy()

            gt_p_box = target['pair_boxes'].cpu().numpy()
            gt_p_act = target['pair_actions'].cpu().numpy()

            score = score[self.action_idx, :, :]
            gt_p_act = gt_p_act[:, self.action_idx]

            self.role_eval1.add_data(hbox, obox, score, gt_h_box, gt_h_act, gt_p_box, gt_p_act)
            self.role_eval2.add_data(hbox, obox, score, gt_h_box, gt_h_act, gt_p_box, gt_p_act)
            self.img_ids.append(img_id)