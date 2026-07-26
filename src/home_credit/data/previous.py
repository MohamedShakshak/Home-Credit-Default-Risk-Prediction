"""Previous application, installments, POS, and credit-card aggregation (fix W3, W4)."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from home_credit.data.loader import (
    load_credit_card,
    load_installments,
    load_pos_cash,
    load_previous_application,
    reduce_memory,
)


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.where(b != 0, a / b, 0.0)


# ── Previous application ──────────────────────────────────────────────


def aggregate_previous_application() -> pd.DataFrame:
    prev = load_previous_application()

    prev["PREV_APP_CREDIT_RATIO"] = _safe_div(
        prev["AMT_APPLICATION"], prev["AMT_CREDIT"].fillna(1),
    )
    prev["PREV_DOWN_PAYMENT_RATE"] = _safe_div(
        prev["AMT_DOWN_PAYMENT"].fillna(0), prev["AMT_GOODS_PRICE"].fillna(1),
    )
    prev["PREV_IS_RECENT"] = (prev["DAYS_DECISION"] >= -365).astype("int8")

    agg_all = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("PREV_APP_CREDIT_RATIO", "mean"),
        PREV_DOWN_PAYMENT_MEAN=("PREV_DOWN_PAYMENT_RATE", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_RECENT_COUNT=("PREV_IS_RECENT", "sum"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    ).reset_index()

    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
    if not approved.empty:
        agg_approved = approved.groupby("SK_ID_CURR").agg(
            PREV_APPROVED_COUNT=("SK_ID_PREV", "count"),
            PREV_APPROVED_CREDIT_MEAN=("AMT_CREDIT", "mean"),
            PREV_APPROVED_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
            PREV_APPROVED_RATIO_MEAN=("PREV_APP_CREDIT_RATIO", "mean"),
        ).reset_index()
    else:
        agg_approved = pd.DataFrame({"SK_ID_CURR": []})

    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"]
    if not refused.empty:
        agg_refused = refused.groupby("SK_ID_CURR").agg(
            PREV_REFUSED_COUNT=("SK_ID_PREV", "count"),
            PREV_REFUSED_CREDIT_MEAN=("AMT_CREDIT", "mean"),
            PREV_REFUSED_DAYS_MEAN=("DAYS_DECISION", "mean"),
        ).reset_index()
    else:
        agg_refused = pd.DataFrame({"SK_ID_CURR": []})

    result = (
        agg_all
        .merge(agg_approved, on="SK_ID_CURR", how="left")
        .merge(agg_refused, on="SK_ID_CURR", how="left")
    )
    result["PREV_APPROVAL_RATE"] = _safe_div(
        result["PREV_APPROVED_COUNT"].fillna(0), result["PREV_APP_COUNT"],
    )
    result["PREV_REFUSAL_RATE"] = _safe_div(
        result["PREV_REFUSED_COUNT"].fillna(0), result["PREV_APP_COUNT"],
    )

    result = reduce_memory(result)
    del prev, approved, refused, agg_all, agg_approved, agg_refused
    gc.collect()
    return result


# ── Installments ──────────────────────────────────────────────────────


def aggregate_installments() -> pd.DataFrame:
    ins = load_installments()

    ins["DAYS_LATE"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DAYS_EARLY"] = -ins["DAYS_LATE"].clip(upper=0)
    ins["DAYS_LATE"] = ins["DAYS_LATE"].clip(lower=0)

    ins["PAYMENT_RATIO"] = _safe_div(
        ins["AMT_PAYMENT"].fillna(0), ins["AMT_INSTALMENT"].fillna(1),
    )
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"].fillna(0)

    ins["IS_LATE"] = (ins["DAYS_LATE"] > 0).astype("int8")
    ins["IS_VERY_LATE"] = (ins["DAYS_LATE"] > 30).astype("int8")
    ins["IS_UNDERPAID"] = (ins["PAYMENT_RATIO"] < 0.95).astype("int8")

    result = ins.groupby("SK_ID_CURR").agg(
        INS_PAYMENT_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INS_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INS_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INS_DAYS_LATE_SUM=("DAYS_LATE", "sum"),
        INS_DAYS_EARLY_MEAN=("DAYS_EARLY", "mean"),
        INS_LATE_COUNT=("IS_LATE", "sum"),
        INS_VERY_LATE_COUNT=("IS_VERY_LATE", "sum"),
        INS_UNDERPAID_COUNT=("IS_UNDERPAID", "sum"),
        INS_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INS_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        INS_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        INS_PAYMENT_DIFF_MAX=("PAYMENT_DIFF", "max"),
        INS_DAYS_ENTRY_MAX=("DAYS_ENTRY_PAYMENT", "max"),
    ).reset_index()

    result["INS_LATE_RATE"] = _safe_div(result["INS_LATE_COUNT"], result["INS_PAYMENT_COUNT"])
    result["INS_VERY_LATE_RATE"] = _safe_div(
        result["INS_VERY_LATE_COUNT"], result["INS_PAYMENT_COUNT"],
    )
    result["INS_UNDERPAID_RATE"] = _safe_div(
        result["INS_UNDERPAID_COUNT"], result["INS_PAYMENT_COUNT"],
    )

    result = reduce_memory(result)
    del ins
    gc.collect()
    return result


# ── POS Cash ──────────────────────────────────────────────────────────


def aggregate_pos_cash() -> pd.DataFrame:
    pos = load_pos_cash()

    pos["IS_DPD"] = (pos["SK_DPD"] > 0).astype("int8")
    pos["IS_DPD_DEF"] = (pos["SK_DPD_DEF"] > 0).astype("int8")

    result = pos.groupby("SK_ID_CURR").agg(
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_SUM=("SK_DPD", "sum"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_DPD_MONTH_COUNT=("IS_DPD", "sum"),
        POS_DPD_DEF_COUNT=("IS_DPD_DEF", "sum"),
        POS_COMPLETED_COUNT=("NAME_CONTRACT_STATUS",
                             lambda x: (x == "Completed").sum()),
        POS_ACTIVE_COUNT=("NAME_CONTRACT_STATUS",
                          lambda x: (x == "Active").sum()),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    ).reset_index()

    result["POS_DPD_RATE"] = _safe_div(
        result["POS_DPD_MONTH_COUNT"], result["POS_MONTHS_COUNT"],
    )
    result["POS_COMPLETED_RATE"] = _safe_div(
        result["POS_COMPLETED_COUNT"], result["POS_MONTHS_COUNT"],
    )

    result = reduce_memory(result)
    del pos
    gc.collect()
    return result


# ── Credit card ───────────────────────────────────────────────────────


def aggregate_credit_card() -> pd.DataFrame:
    cc = load_credit_card()

    cc["CC_UTILIZATION"] = _safe_div(
        cc["AMT_BALANCE"].fillna(0), cc["AMT_CREDIT_LIMIT_ACTUAL"].fillna(1),
    ).clip(0, 1)

    cc["CC_MIN_PAYMENT_RATIO"] = _safe_div(
        cc["AMT_PAYMENT_CURRENT"].fillna(0), cc["AMT_INST_MIN_REGULARITY"].fillna(1),
    )

    cc["CC_DRAWING_TOTAL"] = (
        cc["AMT_DRAWINGS_ATM_CURRENT"].fillna(0)
        + cc["AMT_DRAWINGS_CURRENT"].fillna(0)
        + cc["AMT_DRAWINGS_OTHER_CURRENT"].fillna(0)
        + cc["AMT_DRAWINGS_POS_CURRENT"].fillna(0)
    )
    cc["CC_CASH_DRAW_RATIO"] = _safe_div(
        cc["AMT_DRAWINGS_ATM_CURRENT"].fillna(0),
        cc["CC_DRAWING_TOTAL"].replace(0, np.nan),
    )

    cc["IS_DPD"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["IS_MAXED"] = (cc["CC_UTILIZATION"] > 0.95).astype("int8")

    result = cc.groupby("SK_ID_CURR").agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_UTILIZATION_MEAN=("CC_UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("CC_UTILIZATION", "max"),
        CC_UTILIZATION_MIN=("CC_UTILIZATION", "min"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_PAYMENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
        CC_MIN_PAYMENT_RATIO_MEAN=("CC_MIN_PAYMENT_RATIO", "mean"),
        CC_DRAWING_TOTAL_MEAN=("CC_DRAWING_TOTAL", "mean"),
        CC_CASH_DRAW_RATIO_MEAN=("CC_CASH_DRAW_RATIO", "mean"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_MONTH_COUNT=("IS_DPD", "sum"),
        CC_MAXED_COUNT=("IS_MAXED", "sum"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_LIMIT_MAX=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
    ).reset_index()

    result["CC_DPD_RATE"] = _safe_div(
        result["CC_DPD_MONTH_COUNT"], result["CC_MONTHS_COUNT"],
    )
    result["CC_MAXED_RATE"] = _safe_div(
        result["CC_MAXED_COUNT"], result["CC_MONTHS_COUNT"],
    )

    result = reduce_memory(result)
    del cc
    gc.collect()
    return result


__all__ = [
    "aggregate_credit_card",
    "aggregate_installments",
    "aggregate_pos_cash",
    "aggregate_previous_application",
]
