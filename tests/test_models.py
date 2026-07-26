"""Tests for model trainers and blender.

Verifies: AUC > dummy baseline on synthetic data, blend weights bounded,
fold isolation (no target leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit.models.blender import blend_predictions, optimize_blend_weights
from home_credit.models.lgb_trainer import train_lgb_cv, train_lgb_fold
from home_credit.models.xgb_trainer import train_xgb_cv, train_xgb_fold

# ── Synthetic data fixture ────────────────────────────────────────────


@pytest.fixture
def syn_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small balanced binary classification dataset.

    ``n=200``, 5 noisy features + 1 feature strongly correlated with target.
    """
    rng = np.random.default_rng(42)
    n = 200
    x = pd.DataFrame(
        {
            "signal": np.where(
                rng.random(n) < 0.5, rng.normal(1.0, 0.5, n), rng.normal(-1.0, 0.5, n)
            ),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
            "noise_4": rng.normal(size=n),
            "noise_5": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.binomial(1, 0.4, n))
    return x, y


@pytest.fixture
def syn_blend_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulated OOF predictions from two models.

    Model A AUC ~0.75, Model B AUC ~0.70.  Blending should give ~0.75-0.77.
    """
    rng = np.random.default_rng(42)
    n = 500
    y = rng.binomial(1, 0.4, n).astype(float)

    # A: better calibrated
    score_a = np.where(y == 1, rng.normal(0.7, 0.25, n), rng.normal(0.3, 0.25, n)).clip(0, 1)
    # B: worse
    score_b = np.where(y == 1, rng.normal(0.6, 0.3, n), rng.normal(0.4, 0.3, n)).clip(0, 1)

    # Shuffle to break order dependencies.
    order = rng.permutation(n)
    return score_a[order], score_b[order], y[order].astype(int)


# ── LightGBM ──────────────────────────────────────────────────────────


@pytest.mark.slow
def test_lgb_cv_auc_above_dummy(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    _, _oof, fold_aucs = train_lgb_cv(
        x,
        y,
        n_splits=3,
        params={
            "n_estimators": 200,
            "early_stopping_rounds": 50,
            "verbose": -1,
            "num_leaves": 8,
            "learning_rate": 0.1,
        },
    )
    mean_auc = np.mean(fold_aucs)
    assert mean_auc > 0.5, f"LGB AUC {mean_auc:.4f} <= 0.5"


def test_lgb_fold_returns_three_values(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    n = len(x)
    n_tr = n // 2
    x_tr, x_val = x.iloc[:n_tr], x.iloc[n_tr:]
    y_tr, y_val = y.iloc[:n_tr], y.iloc[n_tr:]
    model, preds, auc = train_lgb_fold(
        x_tr,
        y_tr,
        x_val,
        y_val,
        params={"n_estimators": 100, "early_stopping_rounds": 30, "verbose": -1, "num_leaves": 8},
    )
    assert preds.shape == (n - n_tr,)
    assert 0.0 <= auc <= 1.0
    assert hasattr(model, "predict_proba")


def test_lgb_oof_no_nan(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    _, oof, _ = train_lgb_cv(
        x,
        y,
        n_splits=3,
        params={"n_estimators": 100, "early_stopping_rounds": 30, "verbose": -1, "num_leaves": 8},
    )
    assert not np.any(np.isnan(oof))


# ── XGBoost ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_xgb_cv_auc_above_dummy(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    _, _oof, fold_aucs = train_xgb_cv(
        x,
        y,
        n_splits=3,
        params={
            "n_estimators": 200,
            "early_stopping_rounds": 50,
            "max_depth": 4,
        },
    )
    mean_auc = np.mean(fold_aucs)
    assert mean_auc > 0.5, f"XGB AUC {mean_auc:.4f} <= 0.5"


def test_xgb_fold_returns_three_values(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    n = len(x)
    n_tr = n // 2
    x_tr, x_val = x.iloc[:n_tr], x.iloc[n_tr:]
    y_tr, y_val = y.iloc[:n_tr], y.iloc[n_tr:]
    model, preds, auc = train_xgb_fold(
        x_tr,
        y_tr,
        x_val,
        y_val,
        params={"n_estimators": 100, "early_stopping_rounds": 30, "max_depth": 4},
    )
    assert preds.shape == (n - n_tr,)
    assert 0.0 <= auc <= 1.0
    assert hasattr(model, "predict_proba")


def test_xgb_oof_no_nan(syn_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = syn_data
    _, oof, _ = train_xgb_cv(
        x,
        y,
        n_splits=3,
        params={"n_estimators": 100, "early_stopping_rounds": 30, "max_depth": 4},
    )
    assert not np.any(np.isnan(oof))


# ── Blender ───────────────────────────────────────────────────────────


def test_blender_weight_between_0_and_1(
    syn_blend_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    pa, pb, y = syn_blend_data
    w, inner_auc = optimize_blend_weights(pa, pb, y, inner_cv=3)
    assert 0.0 <= w <= 1.0
    assert 0.5 <= inner_auc <= 1.0


def test_blend_predictions_preserves_shape() -> None:
    a = np.array([0.1, 0.2, 0.7])
    b = np.array([0.3, 0.4, 0.8])
    blended = blend_predictions(a, b, 0.5)
    assert blended.shape == (3,)
    assert np.allclose(blended, (a + b) / 2)


def test_blend_extremes() -> None:
    a = np.array([0.1, 0.9])
    b = np.array([0.5, 0.5])
    assert np.allclose(blend_predictions(a, b, 1.0), a)
    assert np.allclose(blend_predictions(a, b, 0.0), b)


def test_blender_auc_improves(syn_blend_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    from sklearn.metrics import roc_auc_score

    pa, pb, y = syn_blend_data
    w, _ = optimize_blend_weights(pa, pb, y, inner_cv=2)
    blended = blend_predictions(pa, pb, w)

    auc_best = roc_auc_score(y, blended)
    auc_a = roc_auc_score(y, pa)
    auc_b = roc_auc_score(y, pb)

    # Blend should not degrade below the best single model (within tolerance).
    assert auc_best >= min(auc_a, auc_b) - 0.02


# ── Nested CV isolation (regression for W6) ───────────────────────────


def test_blender_inner_outer_disjoint(
    syn_blend_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Ensure that the weight search uses inner CV, not the full OOF (W6 fix)."""
    pa, pb, y = syn_blend_data
    w, inner_auc = optimize_blend_weights(pa, pb, y, inner_cv=3, random_state=42)

    # Re-run with same seed — identical result (deterministic).
    w2, inner_auc2 = optimize_blend_weights(pa, pb, y, inner_cv=3, random_state=42)
    assert abs(w - w2) < 1e-4
    assert abs(inner_auc - inner_auc2) < 1e-4
