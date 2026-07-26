"""Tests for evaluate module: metrics, drift (NaN-aware), fairness (index-aligned).

Regression tests for notebook bugs W16 (NaN-aware drift) and W12 (join on SK_ID_CURR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit.evaluate.drift import drift_report, ks_drift, psi
from home_credit.evaluate.fairness import fairness_report, merge_on_id
from home_credit.evaluate.metrics import auc, brier, expected_loss, ks_statistic, report

# ── Metrics ───────────────────────────────────────────────────────────


def test_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, p) == 1.0


def test_auc_random() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.6, 0.3, 0.7])
    # Not perfect, but > 0.5 if any signal.
    a = auc(y, p)
    assert 0.0 <= a <= 1.0


def test_brier_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier(y, p) == 0.0


def test_ks_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ks = ks_statistic(y, p)
    assert ks > 0.5


def test_ks_all_same_class() -> None:
    y = np.array([0, 0, 0])
    p = np.array([0.1, 0.2, 0.3])
    assert ks_statistic(y, p) == 0.0


def test_expected_loss_always_correct() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    loss = expected_loss(y, p, threshold=0.5)
    assert loss == 0.0


def test_expected_loss_misses() -> None:
    y = np.array([0, 1])
    p = np.array([0.9, 0.1])  # wrong both ways
    loss = expected_loss(y, p, threshold=0.5, fp_cost=1.0, fn_cost=5.0)
    assert loss > 0.0


def test_report_keys() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = report(y, p)
    for key in ("auc", "brier", "ks", "expected_loss", "confusion"):
        assert key in r


# ── Drift (W16 regression) ────────────────────────────────────────────


def test_psi_same_distribution() -> None:
    ref = np.random.default_rng(42).normal(size=1000)
    cur = ref.copy()
    p, nan_shift = psi(ref, cur)
    assert abs(p) < 0.01  # near zero for identical distributions
    assert nan_shift == 0.0


def test_psi_different_distributions() -> None:
    ref = np.random.default_rng(42).normal(0, 1, 1000)
    cur = np.random.default_rng(99).normal(3, 1, 1000)
    p, _ = psi(ref, cur)
    assert p > 0.1  # clearly different


def test_psi_nan_rate_detected() -> None:
    """W16 regression: NaN-rate shift must be visible."""
    ref = np.array([1.0, 2.0, 3.0, np.nan])
    cur = np.array([1.0, np.nan, np.nan, np.nan])  # 25% → 75% NaN
    _p, nan_shift = psi(ref, cur)
    assert nan_shift > 0.4  # NaN rate shifted by ~0.5


def test_psi_all_nan() -> None:
    ref = np.full(100, np.nan)
    cur = np.full(100, np.nan)
    p, nan_shift = psi(ref, cur)
    assert abs(p) < 0.01
    assert nan_shift == 0.0


def test_ks_drift_identical() -> None:
    ref = np.random.default_rng(42).normal(size=500)
    cur = ref.copy()
    ks, nan_shift = ks_drift(ref, cur)
    assert abs(ks) < 0.05
    assert nan_shift == 0.0


def test_ks_drift_different() -> None:
    ref = np.random.default_rng(42).normal(0, 1, 500)
    cur = np.random.default_rng(99).normal(3, 1, 500)
    ks, _ = ks_drift(ref, cur)
    assert ks > 0.3


def test_drift_report_contains_nan_rate() -> None:
    ref = {"feat_a": np.array([1.0, 2.0, 3.0, np.nan])}
    cur = {"feat_a": np.array([np.nan, np.nan, np.nan, np.nan])}
    r = drift_report(ref, cur)
    assert "feat_a" in r
    assert r["feat_a"]["ref_nan_rate"] == 0.25
    assert r["feat_a"]["cur_nan_rate"] == 1.0
    assert r["feat_a"]["psi_nan_shift"] > 0.5


# ── Fairness (W12 regression) ─────────────────────────────────────────


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "predicted_pd": [0.1, 0.3, 0.5, 0.7, 0.9],
            "TARGET": [0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def demographics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "CODE_GENDER": ["M", "F", "M", "F", "M"],
            "AGE_GROUP": ["young", "young", "old", "old", "old"],
        }
    )


def test_fairness_report_keys(
    predictions: pd.DataFrame,
    demographics: pd.DataFrame,
) -> None:
    r = fairness_report(predictions, demographics, protected_attrs=["CODE_GENDER"])
    assert "CODE_GENDER" in r
    assert "M" in r["CODE_GENDER"]["groups"]
    assert "F" in r["CODE_GENDER"]["groups"]


def test_fairness_report_no_row_mismatch(
    predictions: pd.DataFrame,
    demographics: pd.DataFrame,
) -> None:
    r = fairness_report(predictions, demographics)
    assert r["row_count_mismatch"] is False


def test_fairness_multiple_attrs(
    predictions: pd.DataFrame,
    demographics: pd.DataFrame,
) -> None:
    r = fairness_report(predictions, demographics, protected_attrs=["CODE_GENDER", "AGE_GROUP"])
    assert "CODE_GENDER" in r
    assert "AGE_GROUP" in r


def test_merge_on_id_preserves_rows() -> None:
    left = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"SK_ID_CURR": [1, 3], "gender": ["M", "F"]})
    result = merge_on_id(left, right)
    assert len(result) == 3
    assert result.loc[result["SK_ID_CURR"] == 2, "gender"].isna().all()


def test_merge_on_id_raises_on_duplicates() -> None:
    left = pd.DataFrame({"SK_ID_CURR": [1, 2], "val": [10, 20]})
    right = pd.DataFrame({"SK_ID_CURR": [1, 1], "dup": ["a", "b"]})
    with pytest.raises(ValueError, match="row count"):
        merge_on_id(left, right)
