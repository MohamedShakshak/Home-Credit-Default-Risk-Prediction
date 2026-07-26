"""POST /explain — detailed SHAP explanation for a single applicant."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends

from home_credit.api.deps import get_model
from home_credit.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    ShapReason,
)

router = APIRouter(prefix="/explain", tags=["explain"])


@router.post("", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    model: Any = Depends(get_model),
) -> ExplainResponse:
    df = pd.DataFrame([request.application.model_dump(exclude_none=True)])
    result = model.predict(df, params={"return_shap": True})
    row = result.iloc[0]

    # Build full SHAP list and top-5.
    all_shap: list[ShapReason] = []
    top_reasons: list[ShapReason] = []
    for i in range(1, 6):
        feat = row.get(f"shap_feature_{i}")
        val = row.get(f"shap_value_{i}")
        if feat is not None and val is not None and not pd.isna(val):
            sr = ShapReason(feature=str(feat), shap=float(val))
            all_shap.append(sr)
            top_reasons.append(sr)

    return ExplainResponse(
        SK_ID_CURR=int(row["SK_ID_CURR"]) if pd.notna(row.get("SK_ID_CURR")) else None,
        predicted_pd=float(row["predicted_pd"]),
        raw_score=float(row.get("raw_score", 0.0)) if "raw_score" in row else 0.0,
        base_value=float(row.get("base_value", 0.0)) if "base_value" in row else 0.0,
        top_reasons=top_reasons,
        shap_values=all_shap,
    )
