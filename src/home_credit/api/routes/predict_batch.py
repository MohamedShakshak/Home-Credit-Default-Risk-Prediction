"""POST /predict_batch — batch prediction (list or CSV upload)."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile

from home_credit.api.deps import get_model
from home_credit.api.schemas import (
    BatchRequest,
    BatchResponse,
    PredictionResponse,
    ShapReason,
)

router = APIRouter(prefix="/predict_batch", tags=["predict_batch"])


@router.post("", response_model=BatchResponse)
def predict_batch(
    request: BatchRequest,
    model: Any = Depends(get_model),
) -> BatchResponse:
    records = [r.model_dump(exclude_none=True) for r in request.applications]
    df = pd.DataFrame(records)
    result = model.predict(df)

    predictions = []
    for _, row in result.iterrows():
        reasons = []
        for i in range(1, 6):
            feat = row.get(f"shap_feature_{i}")
            val = row.get(f"shap_value_{i}")
            if feat is not None and val is not None and not pd.isna(val):
                reasons.append(ShapReason(feature=str(feat), shap=float(val)))

        predictions.append(
            PredictionResponse(
                SK_ID_CURR=int(row["SK_ID_CURR"]) if pd.notna(row.get("SK_ID_CURR")) else None,
                predicted_pd=float(row["predicted_pd"]),
                decision=int(row["decision"]),
                top_reasons=reasons,
            )
        )

    return BatchResponse(predictions=predictions, count=len(predictions))


@router.post("/csv", response_model=BatchResponse)
async def predict_batch_csv(
    file: UploadFile,
    model: Any = Depends(get_model),
) -> BatchResponse:
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    result = model.predict(df)

    predictions = []
    for _, row in result.iterrows():
        reasons = []
        for i in range(1, 6):
            feat = row.get(f"shap_feature_{i}")
            val = row.get(f"shap_value_{i}")
            if feat is not None and val is not None and not pd.isna(val):
                reasons.append(ShapReason(feature=str(feat), shap=float(val)))
        predictions.append(
            PredictionResponse(
                SK_ID_CURR=int(row["SK_ID_CURR"]) if pd.notna(row.get("SK_ID_CURR")) else None,
                predicted_pd=float(row["predicted_pd"]),
                decision=int(row["decision"]),
                top_reasons=reasons,
            )
        )

    return BatchResponse(predictions=predictions, count=len(predictions))
