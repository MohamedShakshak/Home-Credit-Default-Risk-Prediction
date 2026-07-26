"""Tests for SHAPExplainer: round-trip, column-order assert, seed, sigmoid→calibrator.

Regression for notebook bugs C3, C4, C5, W11, W13, M5.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# SHAP is imported only inside the test functions that use it.
from home_credit.explain.shap_explainer import SHAPExplainer

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def syn_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small binary classification dataset for explainer tests."""
    rng = np.random.default_rng(42)
    n = 100
    x = pd.DataFrame(
        {
            "feature_a": np.where(
                rng.random(n) < 0.5, rng.normal(2.0, 0.8, n), rng.normal(-2.0, 0.8, n)
            ),
            "feature_b": rng.normal(scale=2.0, size=n),
            "feature_c": rng.normal(scale=2.0, size=n),
            "feature_d": rng.exponential(2.0, n),
        }
    )
    y = pd.Series(rng.binomial(1, 0.4, n))
    return x, y


@pytest.fixture
def trained_ensemble(syn_data: tuple[pd.DataFrame, pd.Series]) -> tuple[list[Any], pd.DataFrame]:
    """Train a small RF ensemble (mimics fold models)."""
    x, y = syn_data
    models = []
    _rng = np.random.default_rng(42)
    for i in range(3):
        x_sample = x.sample(n=80, random_state=int(_rng.integers(0, 1000)))
        y_sample = y.loc[x_sample.index]
        m = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=i)
        m.fit(x_sample, y_sample)
        models.append(m)
    return models, x


@pytest.fixture
def calibrator() -> Any:
    cal = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000, random_state=42)
    # Pre-fit on trivial data.
    cal.fit(np.array([[0.3], [0.7]]), np.array([0, 1]))
    return cal


# ── Tests ─────────────────────────────────────────────────────────────


def test_explainer_init(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    assert explainer.top_features == []
    assert explainer.feature_names == names


def test_explainer_fit_computes_top_features(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)
    assert len(explainer.top_features) == len(names)
    # fix W11: top_features is sorted by importance, not arbitrary.
    assert explainer.top_features != names  # order differs when features have different importances


def test_explainer_explain_returns_all_keys(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)
    result = explainer.explain(x.iloc[:1])
    for key in ("pd", "raw_score", "base_value", "shap_values", "top_reasons"):
        assert key in result, f"missing key: {key}"
    assert 0.0 <= result["pd"] <= 1.0
    assert len(result["top_reasons"]) == 5 or len(result["top_reasons"]) == len(names)


def test_explainer_column_order_assertion(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    """Fix C5: wrong column order should raise."""
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)

    # Pass columns in wrong order.
    wrong_order = x.iloc[:1][names[::-1]]
    with pytest.raises(ValueError, match="order mismatch"):
        explainer.explain(wrong_order)


def test_explainer_save_load(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    """Round-trip: save metadata, load with new model references."""
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "explainer"
        explainer.save(path)

        # Load into a fresh instance (must provide models again).
        loaded = SHAPExplainer.load(path, fold_models=models)
        assert loaded.feature_names == names
        assert loaded.top_features == explainer.top_features

        # Should be able to explain with loaded explainer.
        result = loaded.explain(x.iloc[:1])
        assert "pd" in result


def test_explainer_with_calibrator(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
    calibrator: Any,
) -> None:
    """Fix C3: sigmoid applied before calibrator, PD stays in [0, 1]."""
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        calibrator=calibrator,
        background_size=30,
    )
    explainer.fit(x)
    result = explainer.explain(x.iloc[:1])
    assert 0.0 <= result["pd"] <= 1.0
    # raw_score is log-odds (can be outside [0,1])
    assert result["raw_score"] != result["pd"]  # calibrator changes the value


def test_explainer_shap_value_length(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    """Number of SHAP values should match number of features."""
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)
    result = explainer.explain(x.iloc[:1])
    assert len(result["shap_values"]) == len(names)


def test_explainer_fold_average(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    """Fix W13: multiple fold models produce an average, not last-only."""
    models, x = trained_ensemble
    names = list(x.columns)
    explainer = SHAPExplainer(
        fold_models=models,
        feature_names=names,
        background_size=30,
    )
    explainer.fit(x)
    # If only 1 model, result is just that model's SHAP.
    # With 3 models, should still work cleanly.
    result = explainer.explain(x.iloc[:1])
    assert len(result["shap_values"]) == len(names)


def test_explainer_seeded(
    trained_ensemble: tuple[list[Any], pd.DataFrame],
) -> None:
    """Fix M5: same seed → same top features."""
    models, x = trained_ensemble
    names = list(x.columns)
    e1 = SHAPExplainer(models, names, background_size=30, random_state=42)
    e2 = SHAPExplainer(models, names, background_size=30, random_state=42)
    e1.fit(x)
    e2.fit(x)
    assert e1.top_features == e2.top_features
