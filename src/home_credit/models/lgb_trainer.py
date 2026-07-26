"""LightGBM trainer with StratifiedKFold CV, early stopping, fold isolation.

Returns per-fold models + OOF predictions.  No target information leaks
across the train/val boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    import pandas as pd

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": 2.15,
    "early_stopping_rounds": 200,
    "verbose": -1,
    "random_state": 42,
}


def train_lgb_cv(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[list[lgb.LGBMClassifier], np.ndarray, list[float]]:
    """Run stratified k-fold CV with LightGBM.

    Returns
    -------
    models : list of LGBMClassifier
        One fitted model per fold.
    oof_preds : ndarray of shape (n,)
        Out-of-fold predicted probabilities for the positive class.
    fold_aucs : list of float
        AUC score per fold.
    """
    merged_params = {**_DEFAULT_PARAMS, **(params or {})}
    early_stopping = merged_params.pop("early_stopping_rounds", 200)
    _ = merged_params.pop("verbose", -1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models: list[lgb.LGBMClassifier] = []
    oof_preds = np.zeros(len(x), dtype=np.float64)
    fold_aucs: list[float] = []

    for _fold_idx, (tr_idx, val_idx) in enumerate(skf.split(x, y)):
        x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**merged_params)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
        )

        preds = np.asarray(model.predict_proba(x_val, num_iteration=model.best_iteration_))[:, 1]
        oof_preds[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        fold_aucs.append(auc)

        models.append(model)

    return models, oof_preds, fold_aucs


def train_lgb_fold(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
) -> tuple[lgb.LGBMClassifier, np.ndarray, float]:
    """Train a single LGBM fold.

    Returns (model, val_preds, val_auc).
    """
    merged_params = {**_DEFAULT_PARAMS, **(params or {})}
    early_stopping = merged_params.pop("early_stopping_rounds", 200)
    _ = merged_params.pop("verbose", -1)

    model = lgb.LGBMClassifier(**merged_params)
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
    )

    preds = np.asarray(model.predict_proba(x_val, num_iteration=model.best_iteration_))[:, 1]
    auc = roc_auc_score(y_val, preds)
    return model, preds, auc


__all__ = ["_DEFAULT_PARAMS", "train_lgb_cv", "train_lgb_fold"]
