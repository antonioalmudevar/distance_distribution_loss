"""
https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
https://github.com/deeplearning-wisc/cider/blob/master/utils/detection_util.py
"""
from typing import Tuple, List, Dict, Any

import numpy as np
import sklearn.metrics as sk
from scipy import stats
import torch
from torch import Tensor

from .distances import cosine_dist


def _to_native(x):
    return x.item() if str(type(x)).split()[1][1:6]=="numpy" else x


#====================ACCURACY=========================
def calc_accuracy(
        preds: Tensor, 
        target: Tensor, 
        topk: Tuple[int,...]=(1,),
    ) -> List[float]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    return res


def calc_accuracy_np(
        preds: np.ndarray, 
        target: np.ndarray, 
        topk: Tuple[int,...]=(1,),
    ) -> List[float]:
    return sk.accuracy_score(np.argmax(target,1), np.argmax(preds, 1))
    #return calc_accuracy(torch.from_numpy(preds), torch.from_numpy(target), topk)[0]


#====================ECE=========================
def calc_ece(
        preds: Tensor, 
        target: Tensor, 
        n_bins: int=10,
    ) -> float:
    preds = preds.numpy()
    target = target.numpy()
    if target.ndim > 1:
        target = np.argmax(target, axis=1)
    preds_index = np.argmax(preds, axis=1)
    preds_value = []
    for i in range(preds.shape[0]):
        preds_value.append(preds[i, preds_index[i]])
    preds_value = np.array(preds_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(preds.shape[0]):
            if preds_value[i] > a and preds_value[i] <= b:
                Bm[m] += 1
                if preds_index[i] == target[i]:
                    acc[m] += 1
                conf[m] += preds_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return _to_native(ece / sum(Bm))


def calc_ece_np(
        preds: np.ndarray, 
        target: np.ndarray, 
        n_bins: int=10,
    ) -> float:
    target = np.argmax(target, 1)
    return calc_ece(
        torch.from_numpy(preds), torch.from_numpy(target), n_bins
    )


#====================ENTROPY AND CROSSENTROPY=========================
def calc_info_theory(
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float]:
    eps = 1e-6
    #labels_onehot = np.eye(preds.shape[1])[labels]
    ce = ((-labels * np.log(preds+eps)).sum(axis=1)).mean()
    ent = ((-preds * np.log(preds+eps)).sum(axis=1)).mean()
    return _to_native(ent), _to_native(ce)


#====================FEATURES DISTANCE=========================
def calc_feat_distance(
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
    cos_dist = cosine_dist(features, features)
    labels = labels[:, None]
    not_eye = np.logical_not(np.eye(labels.shape[0], dtype=bool))
    mask = np.equal(labels, labels.T) != 0
    mask_pos = np.logical_and(mask, not_eye)
    mask_neg = ~mask_pos
    dist_pos = cos_dist[mask_pos]
    dist_neg = cos_dist[mask_neg]
    return {
        'Pos Mean': _to_native(dist_pos.mean()),
        'Neg Mean': _to_native(dist_neg.mean()),
        'Pos Std':  _to_native(dist_pos.std()),
        'Neg Std':  _to_native(dist_neg.std()),
        'Pos Skew': _to_native(stats.skew(dist_pos)),
        'Neg Skew': _to_native(stats.skew(dist_neg)),
        'Pos Kurt': _to_native(stats.kurtosis(dist_pos, fisher=False)),
        'Neg Kurt': _to_native(stats.kurtosis(dist_neg, fisher=False)),
    }


#====================AUC, AUPR, FPR=========================
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_ood_measures(_pos, _neg, recall_level=0.95) -> Dict[str, Any]:
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return {
        'AUC':  _to_native(auroc), 
        'AUPR': _to_native(aupr), 
        'FPR':  _to_native(fpr),
    }