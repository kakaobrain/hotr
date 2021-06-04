import torch
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_overlap(a, b):
    if type(a) == torch.Tensor:
        if len(a.shape) == 2:
            area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

            iw = torch.min(a[:, 2].unsqueeze(dim=1), b[:, 2]) - torch.max(a[:, 0].unsqueeze(dim=1), b[:, 0])
            ih = torch.min(a[:, 3].unsqueeze(dim=1), b[:, 3]) - torch.max(a[:, 1].unsqueeze(dim=1), b[:, 1])

            iw[iw<0] = 0
            ih[ih<0] = 0

            ua = torch.unsqueeze((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), dim=1) + area - iw * ih
            ua[ua < 1e-8] = 1e-8

            intersection = iw * ih

            return intersection / ua

    elif type(a) == np.ndarray:
        if len(a.shape) == 2:
            area = np.expand_dims((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), axis=0) #(1, K)

            iw = np.minimum(np.expand_dims(a[:, 2], axis=1), np.expand_dims(b[:, 2], axis=0)) \
                - np.maximum(np.expand_dims(a[:, 0], axis=1), np.expand_dims(b[:, 0], axis=0)) \
                + 1
            ih = np.minimum(np.expand_dims(a[:, 3], axis=1), np.expand_dims(b[:, 3], axis=0)) \
                - np.maximum(np.expand_dims(a[:, 1], axis=1), np.expand_dims(b[:, 1], axis=0)) \
                + 1

            iw[iw<0] = 0 # (N, K)
            ih[ih<0] = 0 # (N, K)

            intersection = iw * ih

            ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), axis=1) + area - intersection
            ua[ua < 1e-8] = 1e-8

            return intersection / ua

        elif len(a.shape) == 1:
            area = np.expand_dims((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), axis=0) #(1, K)

            iw = np.minimum(np.expand_dims([a[2]], axis=1), np.expand_dims(b[:, 2], axis=0)) \
                - np.maximum(np.expand_dims([a[0]], axis=1), np.expand_dims(b[:, 0], axis=0))
            ih = np.minimum(np.expand_dims([a[3]], axis=1), np.expand_dims(b[:, 3], axis=0)) \
                - np.maximum(np.expand_dims([a[1]], axis=1), np.expand_dims(b[:, 1], axis=0))

            iw[iw<0] = 0 # (N, K)
            ih[ih<0] = 0 # (N, K)

            ua = np.expand_dims([(a[2] - a[0] + 1) * (a[3] - a[1] + 1)], axis=1) + area - iw * ih
            ua[ua < 1e-8] = 1e-8

            intersection = iw * ih

            return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
    # Returns
            The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap