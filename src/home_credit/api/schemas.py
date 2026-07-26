"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicationRequest(BaseModel):
    """Single applicant's raw data (subset of ``application_train`` columns).

    For brevity only a few required fields are shown. In production the full
    set of raw fields would be listed here or passed as a raw dict.
    """

    SK_ID_CURR: int | None = None
    AMT_INCOME_TOTAL: float | None = None
    AMT_CREDIT: float | None = None
    AMT_ANNUITY: float | None = None
    AMT_GOODS_PRICE: float | None = None
    CNT_FAM_MEMBERS: int | None = None
    DAYS_BIRTH: int | None = None
    DAYS_EMPLOYED: int | None = None
    DAYS_REGISTRATION: int | None = None
    DAYS_ID_PUBLISH: int | None = None
    DAYS_LAST_PHONE_CHANGE: int | None = None
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None
    DEF_30_CNT_SOCIAL_CIRCLE: float | None = None
    DEF_60_CNT_SOCIAL_CIRCLE: float | None = None
    OBS_30_CNT_SOCIAL_CIRCLE: float | None = None
    REGION_RATING_CLIENT: int | None = None
    REGION_RATING_CLIENT_W_CITY: int | None = None
    FLAG_MOBIL: int | None = None
    FLAG_EMP_PHONE: int | None = None
    FLAG_WORK_PHONE: int | None = None
    FLAG_CONT_MOBILE: int | None = None
    FLAG_PHONE: int | None = None
    FLAG_EMAIL: int | None = None
    CNT_CHILDREN: int | None = None
    OWN_CAR_AGE: float | None = None
    CODE_GENDER: str | None = None
    OCCUPATION_TYPE: str | None = None


class ShapReason(BaseModel):
    feature: str = Field(description="Feature name")
    shap: float = Field(description="SHAP value (log-odds contribution)")


class PredictionResponse(BaseModel):
    SK_ID_CURR: int | None = None
    predicted_pd: float = Field(ge=0, le=1, description="Calibrated probability of default")
    decision: int = Field(ge=0, le=1, description="Hard classification (0=accept, 1=reject)")
    top_reasons: list[ShapReason] = Field(default_factory=list, description="Top-5 SHAP reasons")


class BatchRequest(BaseModel):
    applications: list[ApplicationRequest] = Field(min_length=1, max_length=10_000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ExplainRequest(BaseModel):
    application: ApplicationRequest


class ExplainResponse(BaseModel):
    SK_ID_CURR: int | None = None
    predicted_pd: float
    raw_score: float = Field(description="Ensemble log-odds before calibrator")
    base_value: float = Field(description="Expected log-odds (SHAP intercept)")
    top_reasons: list[ShapReason] = Field(description="Top-5 SHAP reasons")
    shap_values: list[ShapReason] | None = Field(default=None, description="All SHAP values")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_stage: str | None = None
