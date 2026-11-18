import numpy as np
from typing import Dict, Tuple, Union, Optional, List, Union
import torch
from sklearn.metrics import (
    auc,
)
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        score_min=0.0,
        score_max=1.0,
        num_ths=100,
        thresholds: Optional[np.ndarray] = None,
        device_id: Optional[Union[int, str]] = None,
    ):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = np.linspace(score_min, score_max, num_ths)[1:]

        if device_id:
            self.device = torch.device(int(device_id))
        else:
            self.device = torch.device("cpu")

        self.reset()

    def update(self, y_true_batch, y_nscore_batch, rois: Optional[list] = None):

        if rois is not None:
            y_true_batch = _apply_rois(y_true_batch, rois)
            y_nscore_batch = _apply_rois(y_nscore_batch, rois)

        fps, tps, fns, tns = _compute_confusion_matrices(
            y_true_batch, y_nscore_batch, self.thresholds, self.device
        )

        self.fps += fps
        self.tps += tps
        self.fns += fns
        self.tns += tns

    def reset(self):
        self.fps = np.zeros_like(self.thresholds, dtype=np.int_)
        self.tps = np.zeros_like(self.thresholds, dtype=np.int_)
        self.fns = np.zeros_like(self.thresholds, dtype=np.int_)
        self.tns = np.zeros_like(self.thresholds, dtype=np.int_)

    def compute_metrics(self, reset: bool = False) -> Dict[str, Union[float, int]]:

        # auroc
        fpr, tpr = _compute_roc_curve(self.fps, self.tps, self.fns, self.tns)
        auroc = auc(fpr, tpr)

        # ap (average precision)
        precisions, recalls = _compute_precision_recall_curve(
            self.fps, self.tps, self.fns
        )
        ap = _compute_average_precision(precisions, recalls)

        # f1 score max
        f1_score_max = _compute_f1_score_max(precisions, recalls)

        assert np.isfinite(auroc).item()
        assert np.isfinite(ap).item()
        assert np.isfinite(f1_score_max).item()

        metrics = {"auroc": auroc, "ap": ap, "f1_score_max": f1_score_max}

        if reset:
            self.reset()

        return metrics

    def compute_fbeta_scores(self, beta=1.0, reset: bool = False):

        precisions, recalls = _compute_precision_recall_curve(
            self.fps, self.tps, self.fns
        )

        fbeta_scores = _compute_fbeta_scores(precisions, recalls, beta=beta)

        assert len(fbeta_scores) == len(self.thresholds)
        assert np.all(np.isfinite(fbeta_scores)).item()

        if reset:
            self.reset()

        return fbeta_scores


def _compute_confusion_matrices(
    y_true: np.ndarray,  # binary
    y_score: np.ndarray,  # binary
    thresholds: np.ndarray,
    device: torch.device,
) -> List[np.ndarray]:

    mask_gt = torch.from_numpy(y_true).bool().to(device)
    prob_map = torch.from_numpy(y_score).float().to(device)

    confusion_matrices = [[] for _ in range(len(thresholds))]
    for n, th in tqdm(enumerate(thresholds), desc="Computing confusion matrices..."):
        _mask_pred = (prob_map > th).bool()

        confusion_matrices[n] = _compute_confusion_matrix(
            mask_gt=mask_gt,
            mask_pred=_mask_pred,
        ).to("cpu", non_blocking=True)

    confusion_matrices = torch.stack(confusion_matrices, dim=2).numpy()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    tns = confusion_matrices[0, 0]
    fps = confusion_matrices[0, 1]
    fns = confusion_matrices[1, 0]
    tps = confusion_matrices[1, 1]

    return fps, tps, fns, tns


def _compute_confusion_matrix(
    mask_gt: torch.BoolTensor, mask_pred: torch.BoolTensor
) -> torch.Tensor:

    confusion_matrix = torch.zeros((2, 2), dtype=torch.int)

    # No need to check for equality to 1 since masks are boolean
    ap_mask = mask_gt
    ap = ap_mask.sum()
    an = mask_gt.numel() - ap

    pp = mask_pred.sum()

    tp = mask_pred[ap_mask].sum()
    fn = ap - tp
    fp = pp - tp
    tn = an - fp

    confusion_matrix[0, 0] = tn
    confusion_matrix[0, 1] = fp
    confusion_matrix[1, 0] = fn
    confusion_matrix[1, 1] = tp

    return confusion_matrix


def _compute_roc_curve(fps, tps, fns, tns):

    pos_count = tps[0] + fns[0]
    neg_count = fps[0] + tns[0]

    if neg_count:
        fpr = fps / neg_count
    else:
        fpr = np.zeros_like(fps)

    if pos_count:
        tpr = tps / pos_count
    else:
        tpr = np.zeros_like(tps)

    return fpr, tpr


def _compute_precision_recall_curve(
    fps: np.ndarray, tps: np.ndarray, fns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    fps = fps.astype(np.float_)
    tps = tps.astype(np.float_)
    fns = fns.astype(np.float_)

    pps = tps + fps

    precisions = np.zeros_like(tps)
    np.divide(tps, pps, out=precisions, where=(pps != 0))

    aps = tps + fns
    recalls = np.zeros_like(tps)
    np.divide(tps, aps, out=recalls, where=(aps != 0))

    return precisions, recalls


def _compute_average_precision(precisions, recalls):
    # FIXME: cover edge cases
    return -np.sum(np.diff(recalls) * np.array(precisions)[:-1])


def _compute_f1_score_max(precisions, recalls) -> float:
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def _compute_fbeta_scores(precisions: np.ndarray, recalls: np.ndarray, beta: float):

    fbeta_scores = np.zeros_like(precisions)

    _numerators = (1 + beta**2) * (precisions * recalls)
    _denominators = beta**2 * precisions + recalls

    np.divide(_numerators, _denominators, out=fbeta_scores, where=_denominators != 0)

    return fbeta_scores

def _apply_rois(y_batch: np.ndarray, rois):

    assert y_batch.ndim == 3

    y_batch_roi = []
    for roi in rois:
        x0, y0, x1, y1 = roi
        _y_batch = y_batch[:, y0:y1, x0:x1].ravel()
        y_batch_roi.append(_y_batch)
    
    y_batch_roi = np.concatenate(y_batch_roi)

    return y_batch_roi