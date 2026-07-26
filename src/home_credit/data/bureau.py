"""Bureau and bureau_balance aggregation (fix W3, W4, M1)."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from home_credit.data.loader import load_bureau, load_bureau_balance, reduce_memory


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.where(b != 0, a / b, 0.0)


def aggregate_bureau() -> pd.DataFrame:
    bur = load_bureau()

    bur["CREDIT_ACTIVE_BINARY"] = (bur["CREDIT_ACTIVE"] == "Active").astype("int8")
    bur["CREDIT_OVERDUE_BINARY"] = (bur["CREDIT_DAY_OVERDUE"] > 0).astype("int8")

    # Fix W3: sentinel 365243 → NaN BEFORE duration arithmetic.
    if "DAYS_CREDIT_ENDDATE" in bur.columns:
        bur["DAYS_CREDIT_ENDDATE"] = bur["DAYS_CREDIT_ENDDATE"].replace(365243, np.nan)

    bur["BUREAU_CREDIT_UTILIZATION"] = _safe_div(
        bur["AMT_CREDIT_SUM_DEBT"].fillna(0),
        bur["AMT_CREDIT_SUM"].fillna(1),
    )
    bur["BUREAU_CREDIT_DURATION"] = bur["DAYS_CREDIT_ENDDATE"] - bur["DAYS_CREDIT"]

    agg_all = bur.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE_BINARY", "sum"),
        BUREAU_OVERDUE_COUNT=("CREDIT_OVERDUE_BINARY", "sum"),
        BUREAU_PROLONG_COUNT=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_DEBT_MAX=("AMT_CREDIT_SUM_DEBT", "max"),
        BUREAU_OVERDUE_AMT_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_OVERDUE_AMT_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_UTILIZATION_MEAN=("BUREAU_CREDIT_UTILIZATION", "mean"),
        BUREAU_UTILIZATION_MAX=("BUREAU_CREDIT_UTILIZATION", "max"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
    ).reset_index()

    active = bur[bur["CREDIT_ACTIVE"] == "Active"]
    if not active.empty:
        agg_active = active.groupby("SK_ID_CURR").agg(
            BUREAU_ACTIVE_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_ACTIVE_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
            BUREAU_ACTIVE_CREDIT_MEAN=("AMT_CREDIT_SUM", "mean"),
            BUREAU_ACTIVE_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_ACTIVE_UTIL_MEAN=("BUREAU_CREDIT_UTILIZATION", "mean"),
        ).reset_index()
    else:
        agg_active = pd.DataFrame({"SK_ID_CURR": []})

    result = agg_all.merge(agg_active, on="SK_ID_CURR", how="left")
    result["BUREAU_ACTIVE_RATIO"] = _safe_div(
        result["BUREAU_ACTIVE_COUNT"], result["BUREAU_LOAN_COUNT"]
    )
    result["BUREAU_OVERDUE_RATIO"] = _safe_div(
        result["BUREAU_OVERDUE_COUNT"], result["BUREAU_LOAN_COUNT"]
    )
    result["BUREAU_DEBT_CREDIT_RATIO"] = _safe_div(
        result["BUREAU_DEBT_TOTAL"], result["BUREAU_CREDIT_SUM_TOTAL"]
    )

    result = reduce_memory(result)
    del bur, active, agg_all, agg_active
    gc.collect()
    return result


def aggregate_bureau_balance() -> pd.DataFrame:
    bb = load_bureau_balance()

    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bb["STATUS_NUM"] = bb["STATUS"].map(status_map).fillna(0).astype("int8")
    bb["IS_DPD"] = (bb["STATUS_NUM"] > 0).astype("int8")
    bb["IS_BAD"] = (bb["STATUS_NUM"] >= 2).astype("int8")

    stats = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_STATUS_MEAN=("STATUS_NUM", "mean"),
        BB_STATUS_MAX=("STATUS_NUM", "max"),
        BB_DPD_COUNT=("IS_DPD", "sum"),
        BB_BAD_COUNT=("IS_BAD", "sum"),
    ).reset_index()

    stats["BB_DPD_RATE"] = _safe_div(stats["BB_DPD_COUNT"], stats["BB_MONTHS_COUNT"])
    stats["BB_BAD_RATE"] = _safe_div(stats["BB_BAD_COUNT"], stats["BB_MONTHS_COUNT"])

    # Need SK_ID_CURR via SK_ID_BUREAU mapping.
    bureau_ids = load_bureau_usecols()
    stats = stats.merge(bureau_ids, on="SK_ID_BUREAU", how="left")

    result = stats.groupby("SK_ID_CURR").agg(
        BB_NUM_BUREAU_CREDITS=("SK_ID_BUREAU", "count"),
        BB_STATUS_MEAN_OF_MEANS=("BB_STATUS_MEAN", "mean"),
        BB_STATUS_MAX_EVER=("BB_STATUS_MAX", "max"),
        BB_TOTAL_DPD_MONTHS=("BB_DPD_COUNT", "sum"),
        BB_TOTAL_BAD_MONTHS=("BB_BAD_COUNT", "sum"),
        BB_DPD_RATE_MEAN=("BB_DPD_RATE", "mean"),
        BB_BAD_RATE_MEAN=("BB_BAD_RATE", "mean"),
        BB_DPD_RATE_MAX=("BB_DPD_RATE", "max"),
    ).reset_index()

    result = reduce_memory(result)
    del bb, stats, bureau_ids
    gc.collect()
    return result


def load_bureau_usecols() -> pd.DataFrame:
    """Load only the mapping columns from bureau to save memory."""
    from home_credit.paths import DATA_DIR

    return pd.read_csv(
        f"{DATA_DIR}/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"]
    )


__all__ = ["aggregate_bureau", "aggregate_bureau_balance"]
