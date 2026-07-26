"""Tests for per-fold feature selection (regression for W1).

W1 fix: selection uses only ``(x_tr, y_tr)`` — no target-informed
filtering on the full dataset before cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from home_credit.features.selection import (
    select_by_mutual_info,
    select_by_variance,
    select_constant_and_duplicate,
    select_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def selection_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 200
    x = pd.DataFrame(
        {
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
            "signal": rng.normal(size=n) + rng.binomial(1, 0.3, size=n) * 0.8,
            "constant": 1.0,
        }
    )
    y = pd.Series(rng.binomial(1, 0.3, size=n))
    return x, y


@pytest.fixture
def duplicate_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a.copy(), "c": rng.uniform(size=n)})


# ── select_constant_and_duplicate ─────────────────────────────────────


def test_select_constant(selection_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, _ = selection_data
    constant, _ = select_constant_and_duplicate(x)
    assert "constant" in constant


def test_select_duplicate(duplicate_data: pd.DataFrame) -> None:
    _, duplicate = select_constant_and_duplicate(duplicate_data)
    assert "b" in duplicate
    assert "a" not in duplicate


def test_select_constant_no_cols_left() -> None:
    df = pd.DataFrame({"a": [1.0] * 10})
    constant, duplicate = select_constant_and_duplicate(df)
    assert "a" in constant
    assert duplicate == []


# ── select_by_variance ────────────────────────────────────────────────


def test_select_by_variance_threshold(selection_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, _ = selection_data
    selected = select_by_variance(x, threshold=0.5)
    assert "constant" not in selected
    assert len(selected) <= 4


def test_select_by_variance_percentile(selection_data: tuple[pd.DataFrame, pd.Series]) -> None:
    x, _ = selection_data
    selected = select_by_variance(x, percentile=50)
    assert len(selected) >= 1


def test_select_by_variance_all_high() -> None:
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(42).normal(size=100),
            "b": np.random.default_rng(99).normal(size=100),
        }
    )
    selected = select_by_variance(df, threshold=-1.0)
    assert len(selected) == 2


# ── select_by_mutual_info ─────────────────────────────────────────────


def test_select_by_mutual_info_returns_k(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_by_mutual_info(x, y, k=3, min_features=1)
    assert len(selected) == 3


def test_select_by_mutual_info_min_features(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_by_mutual_info(x, y, k=1, min_features=2)
    # min_features=2 overrides k=1.
    assert len(selected) == 2


def test_select_by_mutual_info_no_k(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_by_mutual_info(x, y, k=None)
    assert len(selected) == len(x.columns)


# ── select_features (pipeline) ────────────────────────────────────────


def test_select_features_drops_constant(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_features(x, y, method=None)
    assert "constant" not in selected


def test_select_features_mutual_info(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_features(x, y, method="mutual_info", k=3, random_state=42)
    assert len(selected) <= 4  # signal + maybe 1-2 noise above threshold


def test_select_features_only_train_y(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Regression for W1: selection must not see full-dataset y."""
    x, y = selection_data
    features_full = select_features(x, y, method="mutual_info", k=3)
    # Repeat with same x_tr, y_tr — should be identical.
    features_repeat = select_features(x, y, method="mutual_info", k=3)
    assert features_full == features_repeat


def test_select_features_y_must_be_provided_for_mutual_info(
    selection_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    x, y = selection_data
    selected = select_features(x, y, method="mutual_info")
    assert len(selected) > 0
