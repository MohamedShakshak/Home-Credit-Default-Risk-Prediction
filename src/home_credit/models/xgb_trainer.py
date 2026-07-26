"""XGBoost trainer with native NaN handling (fix W7) and tuned scale_pos_weight (fix W8).

Does NOT call ``fillna(-999)`` — XGBoost handles ``NaN`` natively.
``scale_pos_weight`` default = 2.5 (notebook's 11 was over-aggressive).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    import pandas as pd


_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 2.5,
    "early_stopping_rounds": 200,
    "verbosity": 1,
    "enable_categorical": False,
    "random_state": 42,
}


def train_xgb_cv(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[list[xgb.XGBClassifier], np.ndarray, list[float]]:
    """Run stratified k-fold CV with XGBoost (native NaN).

    Returns
    -------
    models : list of XGBClassifier
    oof_preds : ndarray of shape (n,)
    fold_aucs : list of float
    """
    merged_params = {**_DEFAULT_PARAMS, **(params or {})}
    _ = merged_params.pop("verbosity", 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models: list[xgb.XGBClassifier] = []
    oof_preds = np.zeros(len(x), dtype=np.float64)
    fold_aucs: list[float] = []

    for _fold_idx, (tr_idx, val_idx) in enumerate(skf.split(x, y)):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**merged_params)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )

        preds = model.predict_proba(x_val, iteration_range=(0, model.best_iteration + 1))[:, 1]
        oof_preds[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)
        models.append(model)

    return models, oof_preds, fold_aucs


def train_xgb_fold(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[xgb.XGBClassifier, np.ndarray, float]:
    """Train a single XGBoost fold.

    Returns (model, val_preds, val_auc).
    """
    merged_params = {**_DEFAULT_PARAMS, **(params or {})}

    model = xgb.XGBClassifier(**merged_params)
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    preds = model.predict_proba(x_val, iteration_range=(0, model.best_iteration + 1))[:, 1]
    auc = roc_auc_score(y_val, preds)
    return model, preds, auc


__all__ = ["_DEFAULT_PARAMS", "train_xgb_cv", "train_xgb_fold"]
