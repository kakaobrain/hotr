import numpy as np
from hotr.metrics.utils import _compute_ap, compute_overlap
import pdb

class APAgent(object):
    def __init__(self, act_name, iou_threshold=0.5):
        self.act_name = act_name
        self.iou_threshold = iou_threshold

        self.fp = [np.zeros((0,))] * len(act_name)
        self.tp  = [np.zeros((0,))] * len(act_name)
        self.score = [np.zeros((0,))] * len(act_name)
        self.num_ann = [0] * len(act_name)

    def add_data(self, box, act, cat, i_box, i_act):
        for label in range(len(self.act_name)):
            i_inds = (i_act[:, label] == 1)
            self.num_ann[label] += i_inds.sum()

        n_pred = box.shape[0]
        if n_pred == 0 : return

        ######################
        valid_i_inds = (i_act[:, 0] != -1) # (n_i, ) # both in COCO & V-COCO

        overlaps = compute_overlap(box, i_box) # (n_pred, n_i)
        assigned_input = np.argmax(overlaps, axis=1) # (n_pred, )
        v_inds = valid_i_inds[assigned_input] # (n_pred, )

        n_valid = v_inds.sum()

        if n_valid == 0 : return
        valid_box = box[v_inds]
        valid_act = act[v_inds]
        valid_cat = cat[v_inds]

        ######################
        s = valid_act * np.expand_dims(valid_cat, axis=1) # (n_v, #act)

        for label in range(len(self.act_name)):
            inds = np.argsort(s[:, label])[::-1] # (n_v, )
            self.score[label] = np.append(self.score[label], s[inds, label])

            correct_i_inds = (i_act[:, label] == 1)
            if correct_i_inds.sum() == 0:
                self.tp[label] = np.append(self.tp[label], np.array([0]*n_valid))
                self.fp[label] = np.append(self.fp[label], np.array([1]*n_valid))
                continue

            overlaps = compute_overlap(valid_box[inds], i_box) # (n_v, n_i)
            assigned_input = np.argmax(overlaps, axis=1) # (n_v, )
            max_overlap = overlaps[range(n_valid), assigned_input] # (n_v, )

            iou_inds = (max_overlap > self.iou_threshold) & correct_i_inds[assigned_input] # (n_v, )

            i_nonzero = iou_inds.nonzero()[0]
            i_inds = assigned_input[i_nonzero]
            i_iou = np.unique(i_inds, return_index=True)[1]
            i_tp = i_nonzero[i_iou]

            t = np.zeros(n_valid, dtype=np.uint8)
            t[i_tp] = 1
            f = 1-t

            self.tp[label] = np.append(self.tp[label], t)
            self.fp[label] = np.append(self.fp[label], f)

    def evaluate(self):
        average_precisions = dict()
        for label in range(len(self.act_name)):
            if self.num_ann[label] == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-self.score[label])
            self.fp[label] = self.fp[label][indices]
            self.tp[label] = self.tp[label][indices]

            # compute false positives and true positives
            self.fp[label] = np.cumsum(self.fp[label])
            self.tp[label] = np.cumsum(self.tp[label])

            # compute recall and precision
            recall    = self.tp[label] / self.num_ann[label]
            precision = self.tp[label] / np.maximum(self.tp[label] + self.fp[label], np.finfo(np.float64).eps)

            # compute average precision
            average_precisions[label] = _compute_ap(recall, precision) * 100

        print('\n================== AP (Agent) ===================')
        s, n = 0, 0

        for label in range(len(self.act_name)):
            label_name = "_".join(self.act_name[label].split("_")[1:])
            print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(label_name, average_precisions[label], self.num_ann[label]))
            s += average_precisions[label]
            n += 1

        mAP = s/n
        print('| mAP(agent): {:0.2f}'.format(mAP))
        print('----------------------------------------------------')

        return mAP