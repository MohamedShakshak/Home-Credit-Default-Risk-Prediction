"""POST /predict — single applicant prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends

from home_credit.api.deps import get_model
from home_credit.api.schemas import ApplicationRequest, PredictionResponse, ShapReason

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict_single(
    request: ApplicationRequest,
    model: Any = Depends(get_model),
) -> PredictionResponse:
    df = pd.DataFrame([request.model_dump(exclude_none=True)])
    result = model.predict(df)
    row = result.iloc[0]

    reasons = []
    for i in range(1, 6):
        feat = row.get(f"shap_feature_{i}")
        val = row.get(f"shap_value_{i}")
        if feat is not None and val is not None and not pd.isna(val):
            reasons.append(ShapReason(feature=str(feat), shap=float(val)))

    return PredictionResponse(
        SK_ID_CURR=int(row["SK_ID_CURR"]) if pd.notna(row.get("SK_ID_CURR")) else None,
        predicted_pd=float(row["predicted_pd"]),
        decision=int(row["decision"]),
        top_reasons=reasons,
    )
