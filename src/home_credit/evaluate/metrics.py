"""Metrics for binary classification: AUC, Brier, KS, expected loss, confusion matrix.

All functions accept ``y_true`` (int 0/1) and ``y_pred`` (probabilities in [0, 1])
as 1-D numpy arrays.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_pred))


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(brier_score_loss(y_true, y_pred))


def ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between cumulative
    distributions of positive and negative classes.

    Returns a value in [0, 1]; higher = better separation.
    """
    idx = np.argsort(y_pred)
    y_sorted = y_true[idx]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    cum_pos = np.cumsum(y_sorted).astype(float) / n_pos
    cum_neg = np.cumsum(1 - y_sorted).astype(float) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def expected_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> float:
    """Cost-weighted misclassification loss.

    ``fp_cost`` = cost of a false positive (approve a defaulter).
    ``fn_cost`` = cost of a false negative (reject a good applicant).
    """
    pred_label = (y_pred >= threshold).astype(int)
    _tn, fp, fn, _tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    return float((fp * fp_cost + fn * fn_cost) / max(len(y_true), 1))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    """Return confusion matrix counts as a dict."""
    pred_label = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def report(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict[str, float | dict[str, int]]:
    """Single-call evaluation report."""
    return {
        "auc": auc(y_true, y_pred),
        "brier": brier(y_true, y_pred),
        "ks": ks_statistic(y_true, y_pred),
        "expected_loss": expected_loss(y_true, y_pred, threshold=threshold),
        "confusion": confusion(y_true, y_pred, threshold=threshold),
    }


__all__ = [
    "auc",
    "brier",
    "confusion",
    "expected_loss",
    "ks_statistic",
    "report",
]
