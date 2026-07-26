"""Application-level feature engineering (fix C1, W3, W4, M1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.where(b != 0, a / b, 0.0)


def _one_plus(b: pd.Series) -> pd.Series:
    return b.where(b.notna(), 1.0) + 1


# ── Public functions ──────────────────────────────────────────────────


def add_credit_income_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inc = df["AMT_INCOME_TOTAL"]
    cred = df["AMT_CREDIT"]
    ann = df["AMT_ANNUITY"]
    goods = df["AMT_GOODS_PRICE"]
    fam = df["CNT_FAM_MEMBERS"]

    df["CREDIT_INCOME_RATIO"] = cred / _one_plus(inc)
    df["ANNUITY_INCOME_RATIO"] = ann / _one_plus(inc)
    df["CREDIT_TERM"] = ann / _one_plus(cred)
    df["GOODS_CREDIT_RATIO"] = goods / _one_plus(cred)
    df["INCOME_PER_PERSON"] = inc / _one_plus(fam)
    df["CREDIT_PER_PERSON"] = cred / _one_plus(fam)
    df["AMT_DOWN_PAYMENT_PROXY"] = goods - cred
    return df


def add_age_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age = df["DAYS_BIRTH"] / -365.0
    emp = df["DAYS_EMPLOYED"] / -365.0  # NaN for unemployed (sentinel already → NaN)
    reg = df["DAYS_REGISTRATION"] / -365.0
    id_pub = df["DAYS_ID_PUBLISH"] / -365.0
    phone = df["DAYS_LAST_PHONE_CHANGE"] / -365.0

    df["AGE_YEARS"] = age
    df["EMPLOYED_YEARS"] = emp
    df["REGISTRATION_YEARS"] = reg
    df["ID_PUBLISH_YEARS"] = id_pub
    df["LAST_PHONE_CHANGE_YEARS"] = phone

    df["EMPLOYED_TO_AGE_RATIO"] = emp / (age + 1)
    df["REGISTRATION_TO_AGE"] = reg / (age + 1)
    df["ID_TO_AGE_RATIO"] = id_pub / (age + 1)
    df["AGE_AT_EMPLOYMENT_START"] = age - emp
    return df


def add_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    present = [c for c in ext if c in df.columns]

    df["EXT_SOURCE_MEAN"] = df[present].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[present].std(axis=1)
    df["EXT_SOURCE_MIN"] = df[present].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[present].max(axis=1)
    df["EXT_SOURCE_RANGE"] = df["EXT_SOURCE_MAX"] - df["EXT_SOURCE_MIN"]

    df["EXT_SOURCE_PRODUCT"] = df[present].prod(axis=1, min_count=1)

    # Weighted mean — ext_2 most predictive.
    weights = {c: (0.5 if "1" in c else 2.0 if "2" in c else 1.5) for c in present}
    weighted_sum = sum(df[c].fillna(df["EXT_SOURCE_MEAN"]) * w for c, w in weights.items())
    df["EXT_SOURCE_WEIGHTED"] = weighted_sum / 4.0

    df["EXT_SOURCE_COUNT"] = df[present].notna().sum(axis=1)

    # Pairwise ratios.
    if len(present) >= 2:
        df["EXT_12_RATIO"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_2"] + 1e-5)
        if len(present) == 3:
            df["EXT_23_RATIO"] = df["EXT_SOURCE_2"] / (df["EXT_SOURCE_3"] + 1e-5)
            df["EXT_13_RATIO"] = df["EXT_SOURCE_1"] / (df["EXT_SOURCE_3"] + 1e-5)

    # Missing indicators.
    for col in present:
        df[f"{col}_MISSING"] = df[col].isna().astype("int8")

    df["EXT_MEAN_X_CREDIT_RATIO"] = df["EXT_SOURCE_MEAN"] * (
        df["AMT_CREDIT"] / _one_plus(df["AMT_INCOME_TOTAL"])
    )
    return df


def add_social_document_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Social circle defaults.
    def_30 = df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0)
    def_60 = df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0)
    obs_30 = df["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(1)
    df["SOCIAL_CIRCLE_DEFAULT_TOTAL"] = def_30 + def_60
    df["SOCIAL_CIRCLE_DEFAULT_RATE"] = def_30 / (obs_30 + 1)

    # Region rating.
    df["REGION_RATING_MEAN"] = (df["REGION_RATING_CLIENT"] + df["REGION_RATING_CLIENT_W_CITY"]) / 2
    df["REGION_CITY_DIFF"] = df["REGION_RATING_CLIENT_W_CITY"] - df["REGION_RATING_CLIENT"]

    # Document submission.
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    df["DOCUMENT_MISSING"] = (df["DOCUMENT_COUNT"] == 0).astype("int8")

    # Contact flags.
    contact_cols = [
        c
        for c in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if c in df.columns
    ]
    df["CONTACT_COUNT"] = df[contact_cols].sum(axis=1)

    # Children.
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["HAS_CHILDREN"] = (df["CNT_CHILDREN"] > 0).astype("int8")
    return df


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary missingness flags for known high-signal columns."""
    df = df.copy()
    cols = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_3",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "OWN_CAR_AGE",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
        "DAYS_LAST_PHONE_CHANGE",
    ]
    for col in cols:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype("int8")
    # OWN_CAR_AGE NaN ≈ no car.
    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)
    return df


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sequentially apply all application-level feature engineering steps."""
    df = add_credit_income_ratios(df)
    df = add_age_employment_features(df)
    df = add_ext_source_features(df)
    df = add_social_document_features(df)
    df = add_missing_indicators(df)
    return df


__all__ = [
    "add_age_employment_features",
    "add_credit_income_ratios",
    "add_ext_source_features",
    "add_missing_indicators",
    "add_social_document_features",
    "engineer_application_features",
]
