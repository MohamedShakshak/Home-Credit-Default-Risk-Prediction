"""Tests for the drift detection API endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from home_credit.api.deps import reset_model


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_model()
    yield


@pytest.fixture
def client() -> TestClient:
    from home_credit.api.app import create_app

    app = create_app()
    return TestClient(app)


def _make_drift_csv() -> bytes:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 200),
            "feature_b": rng.normal(5, 2, 200),
            "feature_c": rng.exponential(1, 200),
        }
    )
    return df.to_csv(index=False).encode()


def test_drift_report_identical(client: TestClient) -> None:
    csv = _make_drift_csv()
    resp = client.post(
        "/drift/report",
        files={"ref_file": ("ref.csv", csv, "text/csv"), "cur_file": ("cur.csv", csv, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 3
    # Identical data should have near-zero PSI.
    for feat in body["records"]:
        assert feat["psi"] < 0.01


def test_drift_report_different(client: TestClient) -> None:
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur_df = pd.DataFrame({"x": rng.normal(5, 1, 500)})

    ref_csv = ref_df.to_csv(index=False).encode()
    cur_csv = cur_df.to_csv(index=False).encode()

    resp = client.post(
        "/drift/report",
        files={"ref_file": ("ref.csv", ref_csv, "text/csv"), "cur_file": ("cur.csv", cur_csv, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 1
    assert body["records"][0]["psi"] > 0.1  # clearly drifted


def test_drift_report_json(client: TestClient) -> None:
    ref = {"a": [1.0, 2.0, 3.0], "b": [0.1, 0.2, 0.3]}
    cur = {"a": [1.5, 2.5, 3.5], "b": [0.1, 0.2, 0.3]}
    resp = client.post("/drift/report/json", json={"reference": ref, "current": cur})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] == 2


def test_drift_report_nan_sensitivity(client: TestClient) -> None:
    """NaN-rate shift should be detected (W16 regression)."""
    ref_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})  # 0% NaN
    cur_df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan]})  # 75% NaN

    ref_csv = ref_df.to_csv(index=False).encode()
    cur_csv = cur_df.to_csv(index=False).encode()

    resp = client.post(
        "/drift/report",
        files={"ref_file": ("ref.csv", ref_csv, "text/csv"), "cur_file": ("cur.csv", cur_csv, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    rec = body["records"][0]
    assert rec["ref_nan_rate"] == 0.0
    assert rec["cur_nan_rate"] == 0.75
    assert rec["psi_nan_shift"] > 0.5


def test_drift_report_empty_columns(client: TestClient) -> None:
    """Works with empty column overlap."""
    ref_df = pd.DataFrame({"a": [1.0]})
    cur_df = pd.DataFrame({"b": [2.0]})
    ref_csv = ref_df.to_csv(index=False).encode()
    cur_csv = cur_df.to_csv(index=False).encode()
    resp = client.post(
        "/drift/report",
        files={"ref_file": ("ref.csv", ref_csv, "text/csv"), "cur_file": ("cur.csv", cur_csv, "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["n_features"] == 0
