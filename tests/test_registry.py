"""Tests for MLflow registry client and PyFunc ensemble."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from home_credit.registry.mlflow_client import _flatten

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_minimal_app_df() -> pd.DataFrame:
    """Minimal application row that ``engineer_application_features`` can process."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "AMT_INCOME_TOTAL": [100_000.0],
            "AMT_CREDIT": [500_000.0],
            "AMT_ANNUITY": [50_000.0],
            "AMT_GOODS_PRICE": [400_000.0],
            "CNT_FAM_MEMBERS": [3],
            "DAYS_BIRTH": [-10000],
            "DAYS_EMPLOYED": [-2000],
            "DAYS_REGISTRATION": [-3000],
            "DAYS_ID_PUBLISH": [-2000],
            "DAYS_LAST_PHONE_CHANGE": [-500],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.4],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0],
            "DEF_60_CNT_SOCIAL_CIRCLE": [0],
            "OBS_30_CNT_SOCIAL_CIRCLE": [5],
            "REGION_RATING_CLIENT": [2],
            "REGION_RATING_CLIENT_W_CITY": [2],
            "FLAG_DOCUMENT_2": [1],
            "FLAG_DOCUMENT_3": [0],
            "FLAG_DOCUMENT_4": [1],
            "FLAG_DOCUMENT_5": [0],
            "FLAG_DOCUMENT_6": [1],
            "FLAG_DOCUMENT_7": [0],
            "FLAG_DOCUMENT_8": [1],
            "FLAG_DOCUMENT_9": [0],
            "FLAG_DOCUMENT_10": [1],
            "FLAG_MOBIL": [1],
            "FLAG_EMP_PHONE": [0],
            "FLAG_WORK_PHONE": [0],
            "FLAG_CONT_MOBILE": [1],
            "FLAG_PHONE": [0],
            "FLAG_EMAIL": [0],
            "CNT_CHILDREN": [0],
            "OWN_CAR_AGE": [5.0],
            "CODE_GENDER": ["M"],
            "OCCUPATION_TYPE": ["Laborers"],
        }
    )


# ── MLflow client ─────────────────────────────────────────────────────


def test_flatten_simple() -> None:
    d = {"a": 1, "b": "hello"}
    flat = _flatten(d)
    assert flat == {"a": "1", "b": "hello"}


def test_flatten_nested() -> None:
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    flat = _flatten(d)
    assert flat["a.b"] == "1"
    assert flat["a.c"] == "2"
    assert flat["d"] == "3"


def test_flatten_list() -> None:
    d = {"x": [1, 2, 3]}
    flat = _flatten(d)
    assert "x" in flat
    parsed = json.loads(flat["x"])
    assert parsed == [1, 2, 3]


# ── PyFunc ensemble ───────────────────────────────────────────────────


@pytest.fixture
def mock_artifact_dir() -> Path:
    """Create a minimal artifact directory with required metadata."""
    tmp = Path(tempfile.mkdtemp())
    meta = {
        "feature_names": ["a", "b", "c"],
        "lgb_weight": 0.6,
        "threshold": 0.15,
    }
    with open(tmp / "model_metadata.json", "w") as f:
        json.dump(meta, f)
    return tmp


def test_ensemble_pyfunc_init() -> None:
    from home_credit.registry.pyfunc_ensemble import EnsemblePyFunc

    m = EnsemblePyFunc()
    assert m._threshold == 0.5
    assert m._fold_models == []


def test_ensemble_pyfunc_predict_no_models() -> None:
    from home_credit.registry.pyfunc_ensemble import EnsemblePyFunc

    m = EnsemblePyFunc()
    df = _make_minimal_app_df()
    result = m.predict(None, df)  # type: ignore[arg-type]
    assert "SK_ID_CURR" in result.columns
    assert "predicted_pd" in result.columns


def test_save_ensemble_artifact(mock_artifact_dir: Path) -> None:
    from home_credit.registry.pyfunc_ensemble import save_ensemble_artifact

    out = save_ensemble_artifact(
        mock_artifact_dir / "saved",
        fold_models=[],
        feature_names=["a", "b"],
        lgb_weight=0.5,
    )
    assert (out / "fold_models.pkl").exists()
    assert (out / "model_metadata.json").exists()


def test_ensemble_pyfunc_return_shap_param() -> None:
    """return_shap param should not crash when explainer is None."""
    from home_credit.registry.pyfunc_ensemble import EnsemblePyFunc

    m = EnsemblePyFunc()
    df = _make_minimal_app_df()
    result = m.predict(None, df, params={"return_shap": True})  # type: ignore[arg-type]
    assert "predicted_pd" in result.columns


# ── Smoke ─────────────────────────────────────────────────────────────


def test_train_module_importable() -> None:
    import home_credit.train  # noqa: F401


def test_predict_module_importable() -> None:
    import home_credit.predict  # noqa: F401
