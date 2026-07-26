"""Tests for the FastAPI prediction API.

Uses ``TestClient`` with a monkeypatched model to avoid needing a real
MLflow registry connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from home_credit.api.deps import reset_model
from home_credit.api.schemas import PredictionResponse

# ── Mock model ────────────────────────────────────────────────────────


class _MockModel:
    """Minimal mock that acts like ``mlflow.pyfunc.PyFuncModel.predict``."""

    def predict(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {
                "SK_ID_CURR": df.get("SK_ID_CURR", [None] * n),
                "predicted_pd": [0.3] * n,
                "decision": [0] * n,
                "raw_score": [-0.5] * n,
                "base_value": [-0.2] * n,
                "shap_feature_1": ["AMT_CREDIT"] * n,
                "shap_value_1": [0.15] * n,
                "shap_feature_2": ["EXT_SOURCE_2"] * n,
                "shap_value_2": [-0.10] * n,
            }
        )


def _mock_mlflow_model(*args: Any, **kwargs: Any) -> _MockModel:
    return _MockModel()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_model() -> Iterator[None]:
    reset_model()
    yield
    reset_model()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MODEL_URI", "mock")
    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", _mock_mlflow_model)

    from home_credit.api.app import create_app

    app = create_app()
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")


def test_predict_returns_prediction(client: TestClient) -> None:
    payload = {
        "SK_ID_CURR": 1001,
        "AMT_INCOME_TOTAL": 200_000.0,
        "AMT_CREDIT": 500_000.0,
        "AMT_ANNUITY": 55_000.0,
        "AMT_GOODS_PRICE": 400_000.0,
        "CNT_FAM_MEMBERS": 3,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -3000,
        "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -2500,
        "DAYS_LAST_PHONE_CHANGE": -600,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.4,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 5,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 0,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 0,
        "CNT_CHILDREN": 0,
        "OWN_CAR_AGE": 5.0,
        "CODE_GENDER": "M",
        "OCCUPATION_TYPE": "Laborers",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_pd"] == 0.3
    assert body["decision"] == 0
    assert len(body["top_reasons"]) >= 1


def test_predict_missing_optional_fields(client: TestClient) -> None:
    payload = {"AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 300_000.0}
    resp = client.post("/predict", json=payload)
    # Should still succeed (fields default to None).
    assert resp.status_code == 200


def test_predict_returns_valid_schema(client: TestClient) -> None:
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 500_000.0}
    resp = client.post("/predict", json=payload)
    model = PredictionResponse(**resp.json())
    assert 0.0 <= model.predicted_pd <= 1.0
    assert model.decision in (0, 1)


def test_predict_batch(client: TestClient) -> None:
    payload = {
        "applications": [
            {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 100_000.0, "AMT_CREDIT": 500_000.0},
            {"SK_ID_CURR": 2, "AMT_INCOME_TOTAL": 200_000.0, "AMT_CREDIT": 600_000.0},
        ]
    }
    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty_returns_422(client: TestClient) -> None:
    resp = client.post("/predict_batch", json={"applications": []})
    assert resp.status_code == 422


def test_explain(client: TestClient) -> None:
    payload = {
        "application": {
            "SK_ID_CURR": 42,
            "AMT_INCOME_TOTAL": 150_000.0,
            "AMT_CREDIT": 450_000.0,
            "AMT_ANNUITY": 50_000.0,
            "AMT_GOODS_PRICE": 380_000.0,
            "CNT_FAM_MEMBERS": 3,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -3000,
            "DAYS_REGISTRATION": -4000,
            "DAYS_ID_PUBLISH": -2500,
            "DAYS_LAST_PHONE_CHANGE": -600,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0,
            "OBS_30_CNT_SOCIAL_CIRCLE": 5,
            "REGION_RATING_CLIENT": 2,
            "REGION_RATING_CLIENT_W_CITY": 2,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 0,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 0,
            "FLAG_EMAIL": 0,
            "CNT_CHILDREN": 0,
            "OWN_CAR_AGE": 3.0,
            "CODE_GENDER": "F",
            "OCCUPATION_TYPE": "Core staff",
        }
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_pd" in body
    assert "top_reasons" in body
    assert len(body["top_reasons"]) >= 1


def test_explain_missing_optional(client: TestClient) -> None:
    payload = {
        "application": {
            "AMT_INCOME_TOTAL": 100_000.0,
            "AMT_CREDIT": 500_000.0,
        }
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_pd"] == 0.3


def test_openapi_schema(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "predict" in str(schema["paths"])
    assert "explain" in str(schema["paths"])
    assert "health" in str(schema["paths"])


def test_middleware_injects_request_id(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0
