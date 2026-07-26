"""Load raw Home Credit CSVs, downcast, fix sentinels. Paths from env (fix C1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sentinel for unemployed / no-end-date credits (Home Credit uses 365243).
_SENTINEL = 365243

# Columns known to carry the unemployed sentinel that should be NaN.
_DAYS_SENTINEL_COLS = [
    "DAYS_EMPLOYED",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    "DAYS_ENTRY_PAYMENT",
    "DAYS_INSTALMENT",
]


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smallest valid dtype.

    Fixes notebook M1: uses ``pd.api.types.is_integer_dtype`` instead of
    fragile ``str(col_type)[:3] == 'int'`` which fails on pandas nullable
    ``Int*`` types.
    """
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_bool_dtype(dt):
            df[col] = df[col].astype("int8")
            continue
        if pd.api.types.is_integer_dtype(dt):
            if df[col].isna().any():
                # Can't convert nullable int → numpy int; fall back to float32.
                df[col] = df[col].astype(np.float32)
                continue
            c_min = df[col].min()
            c_max = df[col].max()
            for target in (np.int8, np.int16, np.int32):
                if c_min > np.iinfo(target).min and c_max < np.iinfo(target).max:
                    df[col] = df[col].astype(target)
                    break
            continue
        if pd.api.types.is_float_dtype(dt):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            continue
    return df


def fix_sentinels(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Replace 365243 sentinel with NaN in matching columns."""
    target_cols = [c for c in (cols or _DAYS_SENTINEL_COLS) if c in df.columns]
    for col in target_cols:
        df[col] = df[col].replace(_SENTINEL, np.nan)
    return df


def load_csv(name: str, *, path: str | None = None) -> pd.DataFrame:
    """Load a single raw CSV from ``DATA_DIR``, downcast, fix sentinels."""
    from home_credit.paths import DATA_DIR

    source = path or str(DATA_DIR)
    df = pd.read_csv(f"{source}/{name}")
    df = reduce_memory(df)
    df = fix_sentinels(df)
    return df


def load_application() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``application_train.csv`` and ``application_test.csv``."""
    train = load_csv("application_train.csv")
    test = load_csv("application_test.csv")

    # Fix CODE_GENDER 'XNA' → NaN (rare anonymised value).
    for df in (train, test):
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    return train, test


def load_bureau() -> pd.DataFrame:
    """Load ``bureau.csv``."""
    return load_csv("bureau.csv")


def load_bureau_balance() -> pd.DataFrame:
    """Load ``bureau_balance.csv``."""
    return load_csv("bureau_balance.csv")


def load_previous_application() -> pd.DataFrame:
    """Load ``previous_application.csv``.

    Fix W4: no ``inplace=True`` on column slices.
    """
    df = load_csv("previous_application.csv")
    sentinel_cols = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].replace(_SENTINEL, np.nan)
    return df


def load_installments() -> pd.DataFrame:
    return load_csv("installments_payments.csv")


def load_pos_cash() -> pd.DataFrame:
    return load_csv("POS_CASH_balance.csv")


def load_credit_card() -> pd.DataFrame:
    return load_csv("credit_card_balance.csv")


__all__ = [
    "fix_sentinels",
    "load_application",
    "load_bureau",
    "load_bureau_balance",
    "load_credit_card",
    "load_csv",
    "load_installments",
    "load_pos_cash",
    "load_previous_application",
    "reduce_memory",
]
