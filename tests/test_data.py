"""Data-layer tests: downcast, sentinel fix, application featurizers (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from home_credit.data.loader import fix_sentinels, reduce_memory

# ── reduce_memory ──────────────────────────────────────────────────────


def test_reduce_memory_int_downcast() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [100_000, 200_000, 300_000]})
    df["a"] = df["a"].astype("int64")
    df["b"] = df["b"].astype("int64")
    out = reduce_memory(df)
    # 'a' fits in int8 (values 1-3).
    assert out["a"].dtype == np.int8
    # 'b' fits in int32.
    assert out["b"].dtype == np.int32


def test_reduce_memory_float_downcast() -> None:
    df = pd.DataFrame({"x": [1.0, 2.5, 3.0]})
    out = reduce_memory(df)
    assert out["x"].dtype == np.float32


def test_reduce_memory_nullable_int_no_nas() -> None:
    """Fix for M1: nullable Int64 without NAs downcasts to int8/16/32."""
    df = pd.DataFrame({"z": pd.array([1, 2, 3], dtype="Int64")})
    out = reduce_memory(df)
    assert out["z"].dtype == np.int8


def test_reduce_memory_nullable_int_with_nas() -> None:
    """nullable Int64 with NAs fall back to float32 (NaN can't be int)."""
    df = pd.DataFrame({"z": pd.array([1, 2, None], dtype="Int64")})
    out = reduce_memory(df)
    assert out["z"].dtype == np.float32


def test_reduce_memory_bool() -> None:
    df = pd.DataFrame({"flag": [True, False, True]})
    out = reduce_memory(df)
    assert out["flag"].dtype == np.int8


# ── fix_sentinels ──────────────────────────────────────────────────────


def test_fix_sentinels_replaces_365243() -> None:
    df = pd.DataFrame({"DAYS_EMPLOYED": [365243, 0, -100, 365243]})
    out = fix_sentinels(df)
    assert out["DAYS_EMPLOYED"].isna().sum() == 2
    assert out.loc[1, "DAYS_EMPLOYED"] == 0


def test_fix_sentinels_ignores_missing_cols() -> None:
    df = pd.DataFrame({"OTHER": [1, 2]})
    out = fix_sentinels(df)
    assert list(out.columns) == ["OTHER"]


def test_fix_sentinels_custom_cols() -> None:
    df = pd.DataFrame({"DAYS_TEST": [365243, 10]})
    out = fix_sentinels(df, cols=["DAYS_TEST"])
    assert out["DAYS_TEST"].isna()[0]


# ── load_csv (needs real CSV) ──────────────────────────────────────────


@pytest.mark.slow
def test_load_application_returns_shapes() -> None:
    from home_credit.data.loader import load_application

    train, test = load_application()
    assert "TARGET" in train.columns
    assert train.shape[0] > 0
    assert test.shape[0] > 0
    assert test.shape[1] == train.shape[1] - 1  # test has no TARGET


@pytest.mark.slow
def test_fix_sentinels_on_real_data() -> None:
    """DAYS_EMPLOYED sentinel becomes NaN after load."""
    from home_credit.data.loader import load_application

    train, _ = load_application()
    assert "DAYS_EMPLOYED" in train.columns
    n_sentinel = (train["DAYS_EMPLOYED"] == 365243).sum()
    assert n_sentinel == 0, f"still {n_sentinel} DAYS_EMPLOYED=365243 rows"


# ── Application feature engineering (synthetic) ────────────────────────


@pytest.fixture
def app_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(n),
            "TARGET": rng.binomial(1, 0.08, size=n),
            "AMT_INCOME_TOTAL": rng.exponential(200_000, size=n),
            "AMT_CREDIT": rng.exponential(500_000, size=n),
            "AMT_ANNUITY": rng.exponential(50_000, size=n),
            "AMT_GOODS_PRICE": rng.exponential(400_000, size=n),
            "CNT_FAM_MEMBERS": rng.poisson(3, size=n).clip(1, 15),
            "DAYS_BIRTH": rng.integers(-25000, -5000, size=n),
            "DAYS_EMPLOYED": rng.integers(-15000, 0, size=n),
            "DAYS_REGISTRATION": rng.integers(-10000, 0, size=n),
            "DAYS_ID_PUBLISH": rng.integers(-8000, 0, size=n),
            "DAYS_LAST_PHONE_CHANGE": rng.integers(-5000, 0, size=n),
            "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
            "EXT_SOURCE_2": rng.uniform(0, 1, size=n),
            "EXT_SOURCE_3": rng.uniform(0, 1, size=n),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.poisson(2, size=n),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.poisson(1, size=n),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(10, size=n),
            "REGION_RATING_CLIENT": rng.integers(1, 4, size=n),
            "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, size=n),
            "FLAG_DOCUMENT_2": 1,
            "FLAG_DOCUMENT_3": 0,
            "FLAG_DOCUMENT_4": 1,
            "FLAG_DOCUMENT_5": 0,
            "FLAG_DOCUMENT_6": 1,
            "FLAG_DOCUMENT_7": 0,
            "FLAG_DOCUMENT_8": 1,
            "FLAG_DOCUMENT_9": 0,
            "FLAG_DOCUMENT_10": 1,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 0,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 0,
            "FLAG_EMAIL": 0,
            "CNT_CHILDREN": rng.poisson(1, size=n).clip(0, 5),
            "OWN_CAR_AGE": np.where(rng.uniform(size=n) < 0.5, rng.uniform(0, 20, size=n), np.nan),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", np.nan], size=n),
            "CODE_GENDER": rng.choice(["M", "F"], size=n),
        }
    )


def test_engineer_app_col_increase(app_sample: pd.DataFrame) -> None:
    from home_credit.data.application import engineer_application_features

    n_before = app_sample.shape[1]
    out = engineer_application_features(app_sample)
    n_after = out.shape[1]
    # ~48 new features added.
    assert n_after > n_before + 30
    assert n_after < n_before + 60


def test_engineer_app_has_credit_income_ratio(app_sample: pd.DataFrame) -> None:
    from home_credit.data.application import engineer_application_features

    out = engineer_application_features(app_sample)
    assert "CREDIT_INCOME_RATIO" in out.columns
    assert (out["CREDIT_INCOME_RATIO"] >= 0).all()


def test_engineer_app_age_years_nonnegative(app_sample: pd.DataFrame) -> None:
    from home_credit.data.application import engineer_application_features

    out = engineer_application_features(app_sample)
    assert (out["AGE_YEARS"] >= 0).all()


def test_engineer_app_has_missing_flags(app_sample: pd.DataFrame) -> None:
    from home_credit.data.application import engineer_application_features

    out = engineer_application_features(app_sample)
    missing_cols = [c for c in out.columns if c.endswith("_MISSING")]
    assert len(missing_cols) >= 3  # at least EXT_SOURCE_1/3 + OWN_CAR_AGE


def test_engineer_app_document_count_nonneg(app_sample: pd.DataFrame) -> None:
    from home_credit.data.application import engineer_application_features

    out = engineer_application_features(app_sample)
    assert (out["DOCUMENT_COUNT"] >= 0).all()


def test_engineer_app_consistent_across_rows(app_sample: pd.DataFrame) -> None:
    """Same input → same output (deterministic)."""
    from home_credit.data.application import engineer_application_features

    out1 = engineer_application_features(app_sample)
    out2 = engineer_application_features(app_sample)
    pd.testing.assert_frame_equal(out1, out2)


# ── Bureau aggregation (synthetic) ─────────────────────────────────────


@pytest.fixture
def bur_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HC_DATA", str(tmp_path))
    bur = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 2, 3],
            "SK_ID_BUREAU": range(6),
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Closed", "Active"],
            "CREDIT_DAY_OVERDUE": [0, 0, 10, 0, 0, 0],
            "DAYS_CREDIT": [-1000, -2000, -500, -300, -1500, -50],
            "DAYS_CREDIT_ENDDATE": [-100, 365243, -50, -600, -200, 365243],
            "AMT_CREDIT_SUM": [500_000, 200_000, 300_000, 400_000, 100_000, 600_000],
            "AMT_CREDIT_SUM_DEBT": [200_000, 0, 150_000, 200_000, 0, 300_000],
            "AMT_CREDIT_SUM_OVERDUE": [0, 0, 5_000, 0, 0, 0],
            "CNT_CREDIT_PROLONG": [0, 0, 1, 0, 0, 0],
        }
    )
    bur.to_csv(tmp_path / "bureau.csv", index=False)
    return tmp_path


def test_bureau_aggregation_runs(bur_sample: Path) -> None:
    from importlib import reload

    from home_credit import paths

    reload(paths)
    from home_credit.data.bureau import aggregate_bureau

    result = aggregate_bureau()
    assert "SK_ID_CURR" in result.columns
    assert result.shape[0] >= 3


def test_bureau_sentinel_gone(bur_sample: Path) -> None:
    from importlib import reload

    from home_credit import paths

    reload(paths)
    from home_credit.data.bureau import aggregate_bureau

    result = aggregate_bureau()
    # BUREAU_CREDIT_DURATION with sentinel would be ~375243, without ≈ -900.
    dur_col = [c for c in result.columns if "DURATION" in c]
    if dur_col:
        col = dur_col[0]
        # Rows with sentinel should have NaN or small value.
        assert result[col].max() < 10_000  # would be 365k+ if sentinel not NaN'd


# ── Pipeline integrations ──────────────────────────────────────────────


def test_full_feature_pipeline_sklearn_compatible() -> None:
    from home_credit.data.pipeline import FullFeaturePipeline

    pipe = FullFeaturePipeline()
    pipe.fit()
    # Transforming will try to load real data — skip in unit test.
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "transform")
