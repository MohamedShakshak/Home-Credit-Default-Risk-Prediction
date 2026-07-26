"""POST /drift/report — detect distribution shift between reference and current data.

NaN-aware PSI/KS per feature (fix W16: NaN-rate tracked separately).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile
from pydantic import BaseModel, Field

from home_credit.evaluate.drift import drift_report

router = APIRouter(prefix="/drift", tags=["drift"])


class DriftRecord(BaseModel):
    feature: str
    psi: float = Field(ge=0)
    psi_nan_shift: float = Field(ge=0)
    ks: float = Field(ge=0, le=1)
    ks_nan_shift: float = Field(ge=0)
    ref_nan_rate: float = Field(ge=0, le=1)
    cur_nan_rate: float = Field(ge=0, le=1)


class DriftReport(BaseModel):
    records: list[DriftRecord]
    drifted_features: list[str]
    n_features: int
    n_drifted: int


@router.post("/report", response_model=DriftReport)
def drift_report_endpoint(
    ref_file: UploadFile,
    cur_file: UploadFile,
) -> DriftReport:
    """Upload reference and current CSV files; returns per-feature drift metrics."""
    ref_bytes = ref_file.file.read()
    cur_bytes = cur_file.file.read()
    ref_df = pd.read_csv(BytesIO(ref_bytes))
    cur_df = pd.read_csv(BytesIO(cur_bytes))

    return _compute_drift(ref_df, cur_df)


class DriftPayload(BaseModel):
    reference: dict[str, list[float]]
    current: dict[str, list[float]]


@router.post("/report/json", response_model=DriftReport)
def drift_report_json(payload: DriftPayload) -> DriftReport:
    """Send reference and current data as JSON dicts; returns per-feature drift metrics."""
    ref_df = pd.DataFrame(payload.reference)
    cur_df = pd.DataFrame(payload.current)
    return _compute_drift(ref_df, cur_df)


def _compute_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> DriftReport:
    common = set(ref_df.columns) & set(cur_df.columns)
    ref_dict = {c: np.asarray(ref_df[c].values) for c in common}
    cur_dict = {c: np.asarray(cur_df[c].values) for c in common}

    raw = drift_report(ref_dict, cur_dict)

    records: list[DriftRecord] = []
    drifted: list[str] = []
    for feat, m in raw.items():
        dr = DriftRecord(
            feature=feat,
            psi=round(m["psi"], 6),
            psi_nan_shift=round(m["psi_nan_shift"], 6),
            ks=round(m["ks"], 6),
            ks_nan_shift=round(m["ks_nan_shift"], 6),
            ref_nan_rate=round(m["ref_nan_rate"], 6),
            cur_nan_rate=round(m["cur_nan_rate"], 6),
        )
        records.append(dr)
        if m["psi"] > 0.1 or m["ks"] > 0.2:
            drifted.append(feat)

    records.sort(key=lambda r: r.psi, reverse=True)
    return DriftReport(
        records=records,
        drifted_features=drifted,
        n_features=len(records),
        n_drifted=len(drifted),
    )
