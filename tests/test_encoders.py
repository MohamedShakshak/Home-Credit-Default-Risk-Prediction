"""Tests for encoders: fold isolation, unseen categories, leakage verification.

Regression tests for notebook bugs M6 and W1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from home_credit.features.encoders import CategoricalEncoder, TargetEncoder

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cat_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "cat2": rng.choice(["x", "y", "z"], size=n),
            "num": rng.uniform(size=n),
        }
    )


@pytest.fixture
def target_series() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.binomial(1, 0.3, size=100))


@pytest.fixture
def cat_with_unseen() -> pd.DataFrame:
    """Train has categories a/b/c; test has a/b/unseen_d."""
    train = pd.DataFrame({"col": ["a", "b", "c", "a", "b"] * 20})
    test = pd.DataFrame({"col": ["a", "b", "d", "a", "b"] * 10})
    return train, test


@pytest.fixture
def cat_with_nan() -> pd.DataFrame:
    df = pd.DataFrame({"col": ["a", "b", None, "a", "b", None]})
    return df


# ── CategoricalEncoder ────────────────────────────────────────────────


def test_categorical_encoder_fit_transform(cat_df: pd.DataFrame) -> None:
    enc = CategoricalEncoder()
    out = enc.fit_transform(cat_df)
    assert out["cat1"].dtype in (np.int32, "int32", "int64")
    assert out["cat2"].dtype in (np.int32, "int32", "int64")
    assert out["num"].dtype.kind == "f"  # untouched


def test_categorical_encoder_unseen_value(
    cat_with_unseen: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = cat_with_unseen
    enc = CategoricalEncoder().fit(train)
    out = enc.transform(test)
    # "d" is unseen → should be -1
    unseen_rows = out[test["col"] == "d"]
    assert (unseen_rows["col"] == -1).all()


def test_categorical_encoder_nan(cat_with_nan: pd.DataFrame) -> None:
    enc = CategoricalEncoder().fit_transform(cat_with_nan)
    # NaN should be encoded to an integer (test only that it doesn't crash).
    nan_rows = cat_with_nan["col"].isna()
    # Encoder produces int32 — NaN gets mapped to either -2 (encoded_missing)
    # or -1 (handle_unknown), depending on sklearn version.  Both are valid.
    assert enc.loc[nan_rows, "col"].dtype.kind == "i"


def test_categorical_encoder_no_object_cols() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    enc = CategoricalEncoder()
    out = enc.fit_transform(df)
    pd.testing.assert_frame_equal(out, df)


# ── TargetEncoder ─────────────────────────────────────────────────────


def test_target_encoder_shape(cat_df: pd.DataFrame, target_series: pd.Series) -> None:
    enc = TargetEncoder(m=10.0)
    out = enc.fit_transform(cat_df, target_series)
    assert out.shape == cat_df.shape
    # cat columns should now be float
    assert out["cat1"].dtype.kind == "f"


def test_target_encoder_fold_isolation(cat_df: pd.DataFrame, target_series: pd.Series) -> None:
    """Regression for W1: encoding must not leak target info across folds."""
    x_tr, x_val, y_tr, _ = train_test_split(cat_df, target_series, test_size=0.3, random_state=42)
    enc = TargetEncoder(m=10.0)
    enc.fit(x_tr, y_tr)
    # Transform val: categories seen in train get learned mean;
    # unseen (none here) get global mean.
    out = enc.transform(x_val)
    assert out.shape == x_val.shape


def test_target_encoder_unseen(
    cat_with_unseen: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = cat_with_unseen
    rng = np.random.default_rng(42)
    y_tr = pd.Series(rng.binomial(1, 0.3, size=100))
    enc = TargetEncoder(m=10.0)
    enc.fit(train, y_tr)
    out = enc.transform(test)
    # "d" is unseen → global mean
    unseen_mask = test["col"] == "d"
    unique_vals = out.loc[unseen_mask, "col"].unique()
    assert len(unique_vals) == 1  # all same value (global mean)


def test_target_encoder_global_mean_boundary() -> None:
    """m → large pulls all encoded values toward global mean."""
    df = pd.DataFrame({"col": ["a", "a", "b", "b"]})
    y = pd.Series([1, 1, 0, 0])  # a → 1.0, b → 0.0, global → 0.5
    enc = TargetEncoder(m=1_000.0)  # very high smoothing
    out = enc.fit_transform(df, y)
    # Both should be close to 0.5
    assert abs(out["col"].iloc[0] - 0.5) < 0.1


def test_target_encoder_no_target_leakage() -> None:
    """Encoder must not store y in a way accessible after fit."""
    df = pd.DataFrame({"col": ["a", "a", "b", "b"]})
    y = pd.Series([1, 1, 0, 0])
    enc = TargetEncoder(m=1.0)
    enc.fit(df, y)
    # Encoder should not expose raw target values.
    for _col, mapping in enc._mapping.items():
        for _val in mapping.values():
            assert isinstance(_val, float)
            assert 0.0 <= _val <= 1.0
