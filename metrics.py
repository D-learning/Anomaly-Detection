"""ContCore anomaly evaluation metrics.

All returned metrics are fractions in [0, 1] (NOT percentages).

AUPRO@0.30:
  - FPR uses *all* normal pixels from *all* test images.
  - PRO averages coverage over connected GT anomaly regions (equally).
  - Integrate up to FPR=0.30 and divide the area by 0.30.
  - Uses histogram thresholds for a memory-efficient approximation.

Requires: numpy, scipy, scikit-learn.
"""

import numpy as np
from scipy import ndimage
from sklearn import metrics as sk_metrics


def _as_nhw(array, name):
    """Convert (N,H,W), (N,1,H,W) or single (H,W) to (N,H,W)."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N,H,W) or (N,1,H,W), got {array.shape}")
    return array


def _validate_binary_targets(y, name):
    y = np.asarray(y).reshape(-1)
    vals = np.unique(y)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1; got {vals[:10]}")
    if len(vals) != 2:
        raise ValueError(f"{name} must contain BOTH normal (0) and anomalous (1) labels")
    return y.astype(np.uint8, copy=False)


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """Image AUROC, ROC curve. Expects 1D scores and binary labels."""
    y = _validate_binary_targets(anomaly_ground_truth_labels, "image labels")
    s = np.asarray(anomaly_prediction_weights, dtype=np.float64).reshape(-1)
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("Image scores must match image labels and be finite")
    fpr, tpr, thresholds = sk_metrics.roc_curve(y, s)
    return {
        "auroc": float(sk_metrics.roc_auc_score(y, s)),
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }


def f1_score_max(ground_truth, predictions):
    """Test-set best F1; use for reporting only, NOT as a deployment threshold."""
    y = _validate_binary_targets(ground_truth, "F1 targets")
    s = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("F1 scores must match labels and be finite")
    p, r, thresholds = sk_metrics.precision_recall_curve(y, s)
    if thresholds.size == 0:
        return 0.0
    den = p[:-1] + r[:-1]
    f1 = np.divide(2.0 * p[:-1] * r[:-1], den,
                   out=np.zeros_like(den, dtype=np.float64), where=(den > 0))
    return float(np.max(f1))


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """Global pixel AUROC, ROC, optimal F1 threshold and conditional FPR/FNR."""
    s = _as_nhw(anomaly_segmentations, "anomaly_segmentations").astype(np.float64)
    y = (_as_nhw(ground_truth_masks, "ground_truth_masks") > 0).astype(np.uint8)
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError(f"Invalid pixel scores or shape mismatch: {s.shape} vs {y.shape}")
    scores, labels = s.ravel(), y.ravel()
    _validate_binary_targets(labels, "pixel labels")
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, scores)
    auroc = float(sk_metrics.roc_auc_score(labels, scores))
    precision, recall, pr_thresholds = sk_metrics.precision_recall_curve(labels, scores)
    if pr_thresholds.size == 0:
        optimal_threshold = float("nan")
        opt_fpr = opt_fnr = float("nan")
    else:
        p, r = precision[:-1], recall[:-1]
        f1 = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r)>0)
        i = int(np.argmax(f1))  # index always < len(pr_thresholds)
        optimal_threshold = float(pr_thresholds[i])
        pred = scores >= optimal_threshold
        opt_fpr = float(np.mean(pred[labels == 0]))
        opt_fnr = float(np.mean(~pred[labels == 1]))
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": opt_fpr,
        "optimal_fnr": opt_fnr,
    }


def compute_pro(
    ground_truth_masks,
    anomaly_maps,
    fpr_limit=0.30,
    num_thresholds=4096,
    connectivity=2,
    return_curve=False,
):
    """Return normalized AUPRO@fpr_limit in [0,1].

    IMPORTANT argument order: (ground_truth_masks, anomaly_maps).
    GT: (N,H,W) or (N,1,H,W), values 0/1 or 0/255.
    anomaly_maps: same shape, float scores where higher = more anomalous.

    The PRO curve averages recall per GT connected component, while FPR
    counts all background pixels (including pixels in fully normal images).
    Uses histogram threshold approximation, then interpolates at fpr_limit.
    Increase num_thresholds to check convergence if needed.
    """
    gt = _as_nhw(ground_truth_masks, "ground_truth_masks") > 0
    scores = _as_nhw(anomaly_maps, "anomaly_maps").astype(np.float64, copy=False)
    if gt.shape != scores.shape:
        raise ValueError(f"GT and anomaly map shapes differ: {gt.shape} versus {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("anomaly_maps contains NaN or inf")
    if not (0 < fpr_limit <= 1):
        raise ValueError("fpr_limit must lie in (0,1]")
    if not isinstance(num_thresholds, int) or num_thresholds < 2:
        raise ValueError("num_thresholds must be integer >= 2")
    if connectivity not in (1, 2):
        raise ValueError("connectivity 1=4-neighbor; 2=8-neighbor")

    n_background = int(np.count_nonzero(~gt))
    n_positive = int(np.count_nonzero(gt))
    if n_background == 0 or n_positive == 0:
        raise ValueError("PRO needs both GT anomaly regions and background pixels")

    smin, smax = float(scores.min()), float(scores.max())
    if smin == smax:
        # Constant scores have no discrimination; use diagonal ROC-like
        # interpolation from (0,0) to (1,1), giving fpr_limit/2.
        val = fpr_limit / 2.0
        if return_curve:
            return float(val), {"fpr": np.array([0., fpr_limit]),
                                "pro": np.array([0., fpr_limit]),
                                "num_regions": 0,
                                "constant_score": True}
        return float(val)

    # The last score is included in the final bin by numpy.histogram.
    edges = np.linspace(smin, smax, num_thresholds + 1, dtype=np.float64)
    background_hist = np.zeros(num_thresholds, dtype=np.float64)
    component_recall_hist = np.zeros(num_thresholds, dtype=np.float64)
    n_regions = 0
    structure = ndimage.generate_binary_structure(2, connectivity)

    for gt_i, score_i in zip(gt, scores):
        # FPR from the entire dataset including normal-only images.
        background_hist += np.histogram(score_i[~gt_i], bins=edges)[0]
        cc, n_cc = ndimage.label(gt_i, structure=structure)
        for region_id in range(1, n_cc + 1):
            region_scores = score_i[cc == region_id]
            if region_scores.size == 0:
                continue
            hist = np.histogram(region_scores, bins=edges)[0]
            component_recall_hist += hist / region_scores.size
            n_regions += 1

    if n_regions == 0:
        raise ValueError("No connected anomaly regions were found")

    # Descending thresholds: first > max (zero positives), then each bin left edge.
    # At each threshold, sum histograms from highest score bin downwards.
    fp_counts = np.cumsum(background_hist[::-1], dtype=np.float64)
    region_recall_counts = np.cumsum(component_recall_hist[::-1], dtype=np.float64)
    fpr = np.concatenate(([0.], fp_counts / n_background))
    pro = np.concatenate(([0.], region_recall_counts / n_regions))

    # Keep vertical segments at repeated FPRs. Dropping them changes the
    # interpolation: an entirely inverted prediction would falsely score >0.
    # Because fpr is nondecreasing, keep every point up to fpr_limit and
    # interpolate ONE final point to the exact cutoff.
    last_in = int(np.searchsorted(fpr, fpr_limit, side="right"))
    x = fpr[:last_in].copy()
    y = pro[:last_in].copy()
    if x.size == 0:  # FPR starts at zero, but guard defensively.
        x, y = np.array([0.0]), np.array([0.0])
    if x[-1] < fpr_limit:
        if last_in >= len(fpr):
            raise RuntimeError("Computed PRO curve does not reach fpr_limit")
        next_x, next_y = float(fpr[last_in]), float(pro[last_in])
        if next_x <= x[-1]:
            raise RuntimeError("PRO curve has non-increasing FPR crossing")
        fraction = (fpr_limit - x[-1]) / (next_x - x[-1])
        y_at_limit = y[-1] + fraction * (next_y - y[-1])
        x = np.concatenate((x, [fpr_limit]))
        y = np.concatenate((y, [y_at_limit]))

    area = float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5))
    result = float(np.clip(area / fpr_limit, 0.0, 1.0))
    if return_curve:
        return result, {"fpr": x, "pro": y, "num_regions": n_regions,
                        "num_background_pixels": n_background,
                        "constant_score": False}
    return result


def evaluate_anomaly_detection(scores, segmentations, labels_gt, masks_gt):
    """Seven reporting metrics in [0,1], using all test images/pixels.

    F1 is the maximum over test thresholds, not a calibrated operating point.
    AUPRO is component-macro, histogram approximation at 4096 bins, FPR <= .30,
    with 8-connected regions. The caller converts fractions to percentages.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = _validate_binary_targets(labels_gt, "image labels")
    maps = _as_nhw(segmentations, "segmentations")
    masks = (_as_nhw(masks_gt, "masks_gt") > 0).astype(np.uint8)
    if maps.shape != masks.shape or maps.shape[0] != len(labels):
        raise ValueError("Image labels, anomaly maps and masks must align")
    image_auroc = compute_imagewise_retrieval_metrics(scores, labels)["auroc"]
    pixel_auroc = compute_pixelwise_retrieval_metrics(maps, masks)["auroc"]
    return (image_auroc, pixel_auroc,
            f1_score_max(labels, scores), f1_score_max(masks, maps),
            float(sk_metrics.average_precision_score(labels, scores)),
            float(sk_metrics.average_precision_score(masks.ravel(), maps.ravel())),
            compute_pro(masks, maps, fpr_limit=0.30))
