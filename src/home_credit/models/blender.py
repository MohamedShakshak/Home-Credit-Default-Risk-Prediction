"""Ensemble blender with nested-CV weight search (fix W6: weights tune on inner CV, outer OOF is honest).

Usage::

    weights, inner_auc = optimize_blend_weights(preds_lgb, preds_xgb, y, inner_cv=3)
    blended = blend_predictions(preds_lgb, preds_xgb, weights)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    import pandas as pd


def _auc_loss(
    weights: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, y: np.ndarray
) -> float:
    """Negative AUC (minimize → maximise AUC)."""
    blended = weights[0] * preds_a + (1 - weights[0]) * preds_b
    return float(-roc_auc_score(y, blended))


def optimize_blend_weights(
    preds_lgb: np.ndarray,
    preds_xgb: np.ndarray,
    y: pd.Series | np.ndarray,
    inner_cv: int = 3,
    n_trials: int = 50,
    bounds: tuple[float, float] = (0.0, 1.0),
    random_state: int = 42,
) -> tuple[float, float]:
    """Find the optimal weight for LGB (XGB weight = 1 - LGB weight) using **nested CV**.

    Inner CV folds search for the weight that maximises AUC.  The reported
    ``inner_auc`` is the mean across inner folds — this is NOT the OOF AUC
    that will later be reported; the outer-fold OOF (from training) remains
    the honest estimate.

    Returns
    -------
    lgb_weight : float
        Optimal weight for LGB predictions.
    inner_auc : float
        Mean inner-fold AUC at the optimal weight (optimistic; use outer OOF
        for the honest evaluation).
    """
    y_arr = np.asarray(y)
    _rng = np.random.default_rng(random_state)

    skf_inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

    inner_aucs: list[float] = []
    best_weights: list[float] = []

    for tr_idx, val_idx in skf_inner.split(preds_lgb, y_arr):
        p_a_tr, p_a_val = preds_lgb[tr_idx], preds_lgb[val_idx]
        p_b_tr, p_b_val = preds_xgb[tr_idx], preds_xgb[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        # Bind loop values via default arg to avoid closure late-binding.
        def loss(w: np.ndarray, _pa: np.ndarray = p_a_tr, _pb: np.ndarray = p_b_tr, _y: np.ndarray = y_tr) -> float:
            return _auc_loss(w, _pa, _pb, _y)

        result = minimize(
            loss,
            x0=np.array([0.5]),
            bounds=[bounds],
            method="L-BFGS-B",
            options={"maxiter": n_trials},
        )
        w = float(result.x[0])

        # Evaluate on inner val.
        blended_val = w * p_a_val + (1 - w) * p_b_val
        inner_aucs.append(roc_auc_score(y_val, blended_val))
        best_weights.append(w)

    # Final weight = median of inner-fold best weights.
    lgb_weight = float(np.median(best_weights))
    inner_auc = float(np.mean(inner_aucs))
    return lgb_weight, inner_auc


def blend_predictions(
    preds_lgb: np.ndarray,
    preds_xgb: np.ndarray,
    lgb_weight: float,
) -> np.ndarray:
    """Weighted average: ``lgb_weight * preds_lgb + (1 - lgb_weight) * preds_xgb``."""
    return lgb_weight * preds_lgb + (1 - lgb_weight) * preds_xgb


__all__ = ["blend_predictions", "optimize_blend_weights"]
