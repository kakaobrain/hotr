import numpy as np
import torch
from hotr.metrics.utils import _compute_ap, compute_overlap

class APRole(object):
    def __init__(self, act_name, scenario_flag=True, iou_threshold=0.5):
        self.act_name = act_name
        self.iou_threshold = iou_threshold

        self.scenario_flag = scenario_flag
        # scenario_1 : True
        # scenario_2 : False

        self.fp = [np.zeros((0,))] * len(act_name)
        self.tp  = [np.zeros((0,))] * len(act_name)
        self.score = [np.zeros((0,))] * len(act_name)
        self.num_ann = [0] * len(act_name)

    def add_data(self, h_box, o_box, score, i_box, i_act, p_box, p_act):
        # i_box, i_act : to check if only in COCO
        for label in range(len(self.act_name)):
            p_inds = (p_act[:, label] == 1)
            self.num_ann[label] += p_inds.sum()

        if h_box.shape[0] == 0 : return # if no prediction, just return
        # COCO (O), V-COCO (X) __or__ collater, no ann in image => ignore

        valid_i_inds = (i_act[:, 0] != -1) # (n_i, )
        overlaps = compute_overlap(h_box, i_box) # (n_h, n_i)
        assigned_input = np.argmax(overlaps, axis=1) # (n_h, )
        v_inds = valid_i_inds[assigned_input] # (n_h, )

        h_box = h_box[v_inds]
        score = score[:, v_inds, :]
        if h_box.shape[0] == 0 : return
        n_h = h_box.shape[0]

        valid_p_inds = (p_act[:, 0] != -1) | (p_box[:, 0] != -1)
        p_act = p_act[valid_p_inds]
        p_box = p_box[valid_p_inds]

        n_o = o_box.shape[0]
        if n_o == 0:
            # no prediction for object
            score = score.squeeze(axis=2) # (#act, n_h)

            for label in range(len(self.act_name)):
                h_inds = np.argsort(score[label])[::-1] # (n_h, )
                self.score[label] = np.append(self.score[label], score[label, h_inds])

                p_inds = (p_act[:, label] == 1)
                if p_inds.sum() == 0:
                    self.tp[label] = np.append(self.tp[label], np.array([0]*n_h))
                    self.fp[label] = np.append(self.fp[label], np.array([1]*n_h))
                    continue

                h_overlaps = compute_overlap(h_box[h_inds], p_box[p_inds, :4]) # (n_h, n_p)
                assigned_p = np.argmax(h_overlaps, axis=1) # (n_h, )
                h_max_overlap = h_overlaps[range(n_h), assigned_p] # (n_h, )

                o_overlaps = compute_overlap(np.zeros((n_h, 4)), p_box[p_inds][assigned_p, 4:8])
                o_overlaps = np.diag(o_overlaps) # (n_h, )

                no_role_inds = (p_box[p_inds][assigned_p, 4] == -1) # (n_h, )
                # human (o), action (o), no object in actual image

                h_iou_inds = (h_max_overlap > self.iou_threshold) # (n_h, )
                o_iou_inds = (o_overlaps > self.iou_threshold) # (n_h, )

                # scenario1 is not considered (already no object)
                o_iou_inds[no_role_inds] = 1

                iou_inds = (h_iou_inds & o_iou_inds)
                p_nonzero = iou_inds.nonzero()[0]
                p_inds = assigned_p[p_nonzero]
                p_iou = np.unique(p_inds, return_index=True)[1]
                p_tp = p_nonzero[p_iou]

                t = np.zeros(n_h, dtype=np.uint8)
                t[p_tp] = 1
                f = 1-t

                self.tp[label] = np.append(self.tp[label], t)
                self.fp[label] = np.append(self.fp[label], f)

        else:
            s_obj_argmax = np.argmax(score.reshape(-1, n_o), axis=1).reshape(-1, n_h) # (#act, n_h)
            s_obj_max = np.max(score.reshape(-1, n_o), axis=1).reshape(-1, n_h) # (#act, n_h)

            h_overlaps = compute_overlap(h_box, p_box[:, :4]) # (n_h, n_p)

            for label in range(len(self.act_name)):
                h_inds = np.argsort(s_obj_max[label])[::-1] # (n_h, )
                self.score[label] = np.append(self.score[label], s_obj_max[label, h_inds])

                p_inds = (p_act[:, label] == 1) # (n_p, )
                if p_inds.sum() == 0:
                    self.tp[label] = np.append(self.tp[label], np.array([0]*n_h))
                    self.fp[label] = np.append(self.fp[label], np.array([1]*n_h))
                    continue

                h_overlaps = compute_overlap(h_box[h_inds], p_box[:, :4]) # (n_h, n_p) # match for all hboxes
                h_max_overlap = np.max(h_overlaps, axis=1) # (n_h, ) # get the max overlap for hbox

                # for same human, multiple pairs exist. find the human box that has the same idx with max overlap hbox.
                h_max_temp = np.expand_dims(h_max_overlap, axis=1)
                h_over_thresh = (h_overlaps == h_max_temp) # (n_h, n_p)
                h_over_thresh = h_over_thresh & np.expand_dims(p_inds, axis=0) # (n_h, n_p) # find only for current act

                h_valid = h_over_thresh.sum(axis=1)>0 # (n_h, ) # at least one is True
                # h_valid -> if all is False, then argmax becomes 0. <- prevent
                assigned_p = np.argmax(h_over_thresh, axis=1) # (n_h, ) # p only for current act

                o_mapping_box = o_box[s_obj_argmax[label]][h_inds] # (n_h, ) # find where T is.
                p_mapping_box = p_box[assigned_p, 4:8] # (n_h, 4)

                o_overlaps = compute_overlap(o_mapping_box, p_mapping_box)
                o_overlaps = np.diag(o_overlaps) # (n_h, )
                o_overlaps.setflags(write=1)
                if (~h_valid).sum() > 0:
                    o_overlaps[~h_valid] = 0 # (n_h, )

                no_role_inds = (p_box[assigned_p, 4] == -1) # (n_h, )
                nan_box_inds = np.all(o_mapping_box == 0, axis=1) | np.all(np.isnan(o_mapping_box), axis=1)
                no_role_inds = no_role_inds & h_valid
                nan_box_inds = nan_box_inds & h_valid

                h_iou_inds = (h_max_overlap > self.iou_threshold) # (n_h, )
                o_iou_inds = (o_overlaps > self.iou_threshold) # (n_h, )

                if self.scenario_flag: # scenario_1
                    o_iou_inds[no_role_inds & nan_box_inds] = 1
                    o_iou_inds[no_role_inds & ~nan_box_inds] = 0
                else: # scenario_2
                    o_iou_inds[no_role_inds] = 1

                iou_inds = (h_iou_inds & o_iou_inds)
                p_nonzero = iou_inds.nonzero()[0]
                p_inds = assigned_p[p_nonzero]
                p_iou = np.unique(p_inds, return_index=True)[1]
                p_tp = p_nonzero[p_iou]

                t = np.zeros(n_h, dtype=np.uint8)
                t[p_tp] = 1
                f = 1-t

                self.tp[label] = np.append(self.tp[label], t)
                self.fp[label] = np.append(self.fp[label], f)

    def evaluate(self, print_log=False):
        average_precisions = dict()
        role_num = 1 if self.scenario_flag else 2
        for label in range(len(self.act_name)):

            # sort by score
            indices = np.argsort(-self.score[label])
            self.fp[label] = self.fp[label][indices]
            self.tp[label] = self.tp[label][indices]


            if self.num_ann[label] == 0:
                average_precisions[label] = 0
                continue

            # compute false positives and true positives
            self.fp[label] = np.cumsum(self.fp[label])
            self.tp[label] = np.cumsum(self.tp[label])

            # compute recall and precision
            recall    = self.tp[label] / self.num_ann[label]
            precision = self.tp[label] / np.maximum(self.tp[label] + self.fp[label], np.finfo(np.float64).eps)

            # compute average precision
            average_precisions[label] = _compute_ap(recall, precision) * 100

        if print_log: print(f'\n============= AP (Role scenario_{role_num}) ==============')
        s, n = 0, 0

        for label in range(len(self.act_name)):
            if 'point' in self.act_name[label]:
                continue
            label_name = "_".join(self.act_name[label].split("_")[1:])
            if print_log: print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(label_name, average_precisions[label], self.num_ann[label]))
            if self.num_ann[label] != 0 :
                s += average_precisions[label]
                n += 1

        mAP = s/n
        if print_log:
            print('| mAP(role scenario_{:d}): {:0.2f}'.format(role_num, mAP))
            print('----------------------------------------------------')

        return mAP