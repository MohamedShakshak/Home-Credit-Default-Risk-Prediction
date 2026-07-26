"""Probability calibrator with held-out evaluation split (fix C2, W9).

Fixes from notebooks:

- **C2**: calibrator fit + Brier evaluated on **same** OOF slice → overfits.
  Now ``fit`` splits OOF into ``cal_fit`` / ``cal_eval`` internally.
  Brier reported on ``cal_eval`` only.  The fit/eval split is disjoint
  (``set(cal_fit_idx).isdisjoint(cal_eval_idx)``).

- **W9**: calibrator *method* selection also uses ``cal_eval`` Brier,
  not the same data the calibrator trained on.

Uses ``LogisticRegression`` for sigmoid (Platt) calibration and
``IsotonicRegression`` for non-parametric calibration — avoids deprecated
``CalibratedClassifierCV(cv="prefit")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import pandas as pd


def calibrate_with_split(
    oof_preds: np.ndarray,
    y_oof: pd.Series | np.ndarray,
    *,
    test_size: float = 0.5,
    methods: list[str] | None = None,
    random_state: int = 42,
) -> tuple[list[Any], np.ndarray, float, str]:
    """Split OOF into cal_fit / cal_eval, fit candidate calibrators, pick best.

    Parameters
    ----------
    oof_preds : ndarray of shape (n,)
        Out-of-fold predicted probabilities from the ensemble.
    y_oof : array of shape (n,)
        True binary labels for the OOF set.
    test_size : float
        Fraction of OOF held out for calibrator evaluation.
    methods : list of str, optional
        Calibrator methods to try (default: ``["sigmoid", "isotonic"]``).

    Returns
    -------
    calibrators : list of fitted calibrator objects
        One per method in ``methods``.
    cal_eval_preds : ndarray of shape (n_eval,)
        Predictions from the **best** calibrator on the evaluation split.
    cal_eval_brier : float
        Brier score of the **best** calibrator on the evaluation split
        (honest, not trained on this data).
    best_method : str
        Name of the best-performing method.
    """
    _methods = methods or ["sigmoid", "isotonic"]
    y_arr = np.asarray(y_oof).ravel()

    stratify = y_arr if np.bincount(y_arr.astype(int)).min() > 1 else None

    cal_fit_preds, cal_eval_preds, y_fit, y_eval = train_test_split(
        oof_preds, y_arr, test_size=test_size, stratify=stratify, random_state=random_state,
    )

    best_brier = np.inf
    best_method: str = _methods[0]
    calibrators: list[Any] = []

    for method in _methods:
        cal = _fit_single_calibrator(cal_fit_preds, y_fit, method=method)
        preds_eval = _calibrate_predict(cal, cal_eval_preds, method=method)
        brier = float(brier_score_loss(y_eval, preds_eval))

        calibrators.append(cal)

        if brier < best_brier:
            best_brier = brier
            best_method = method

    best_cal = calibrators[_methods.index(best_method)]
    cal_eval_best_preds = _calibrate_predict(best_cal, cal_eval_preds, method=best_method)

    return calibrators, cal_eval_best_preds, best_brier, best_method


def _fit_single_calibrator(
    cal_preds: np.ndarray,
    y: np.ndarray,
    method: str = "sigmoid",
) -> Any:
    """Fit a single calibrator (sigmoid or isotonic)."""
    if method == "sigmoid":
        cal = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000, random_state=42)
        cal.fit(cal_preds[:, None], y)
        return cal

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
        cal.fit(cal_preds, y)
        return cal

    msg = f"Unknown calibration method: {method}"
    raise ValueError(msg)


def _calibrate_predict(cal: Any, preds: np.ndarray, method: str) -> np.ndarray:
    """Get calibrated probabilities for ``preds``."""
    if method == "sigmoid":
        return np.asarray(cal.predict_proba(preds[:, None])[:, 1])
    if method == "isotonic":
        return np.asarray(cal.predict(preds).clip(0, 1))
    msg = f"Unknown calibration method: {method}"
    raise ValueError(msg)


__all__ = ["calibrate_with_split"]
