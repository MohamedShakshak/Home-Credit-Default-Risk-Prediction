"""Feature engineering utilities: safe column ops, feature groups, composition helpers.

Fixes notebook W2: uses ``.loc`` and ``.assign`` instead of chained ``.iloc``
assignment that triggers ``SettingWithCopyWarning`` in pandas 2.x+.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# ── Safe column assignment ────────────────────────────────────────────


def assign_column(
    df: pd.DataFrame,
    name: str,
    values: pd.Series | Any,
) -> pd.DataFrame:
    """Return a new DataFrame with ``df[name] = values``.

    Uses ``.loc`` under the hood to avoid ``SettingWithCopyWarning``.
    Equivalent to ``df.assign(**{name: values})`` but allows passing
    the target index explicitly for safety.
    """
    return df.assign(**{name: values})


def assign_columns(
    df: pd.DataFrame,
    **kwargs: pd.Series | Any,
) -> pd.DataFrame:
    """Return a new DataFrame with multiple columns assigned via ``.assign``.

    Example::

        df = assign_columns(df, CREDIT_RATIO=a / b, FLAG_X=...)
    """
    return df.assign(**kwargs)


# ── Feature groups (metadata for tracking) ────────────────────────────


# Groups of features produced by the data-layer featurizers.  Each entry
# maps (group_name -> list_of_suffixes_or_exact_columns).  Used for
# ablation studies and logging.
FEATURE_GROUPS: dict[str, list[str]] = {
    "credit_income_ratios": [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_TERM",
        "GOODS_CREDIT_RATIO",
        "INCOME_PER_PERSON",
        "CREDIT_PER_PERSON",
        "AMT_DOWN_PAYMENT_PROXY",
    ],
    "age_employment": [
        "AGE_YEARS",
        "EMPLOYED_YEARS",
        "REGISTRATION_YEARS",
        "ID_PUBLISH_YEARS",
        "LAST_PHONE_CHANGE_YEARS",
        "EMPLOYED_TO_AGE_RATIO",
        "REGISTRATION_TO_AGE",
        "ID_TO_AGE_RATIO",
        "AGE_AT_EMPLOYMENT_START",
    ],
    "ext_source": [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_RANGE",
        "EXT_SOURCE_PRODUCT",
        "EXT_SOURCE_WEIGHTED",
        "EXT_SOURCE_COUNT",
        "EXT_12_RATIO",
        "EXT_23_RATIO",
        "EXT_13_RATIO",
        "EXT_MEAN_X_CREDIT_RATIO",
    ],
    "social_document": [
        "SOCIAL_CIRCLE_DEFAULT_TOTAL",
        "SOCIAL_CIRCLE_DEFAULT_RATE",
        "REGION_RATING_MEAN",
        "REGION_CITY_DIFF",
        "DOCUMENT_COUNT",
        "DOCUMENT_MISSING",
        "CONTACT_COUNT",
        "CHILDREN_RATIO",
        "HAS_CHILDREN",
    ],
    "missing_indicators": [
        # all columns ending with _MISSING
        "_MISSING",
    ],
    "bureau": [
        "BUREAU_",
    ],
    "bureau_balance": [
        "BB_",
    ],
    "previous_application": [
        "PREV_",
    ],
    "installments": [
        "INS_",
    ],
    "pos_cash": [
        "POS_",
    ],
    "credit_card": [
        "CC_",
    ],
}


def get_feature_group(col: str) -> str | None:
    """Return the name of the first matching feature group, or ``None``."""
    for group, patterns in FEATURE_GROUPS.items():
        for pat in patterns:
            if pat.endswith("_") and col.startswith(pat):
                return group
            if pat.startswith("_") and col.endswith(pat):
                return group
            if pat == col:
                return group
    return None


def group_columns(
    columns: list[str],
) -> dict[str, list[str]]:
    """Partition feature names into groups using ``FEATURE_GROUPS``.

    Columns matching no group are placed under the key ``"other"``.
    """
    result: dict[str, list[str]] = {}
    for col in columns:
        g = get_feature_group(col)
        if g is None:
            result.setdefault("other", []).append(col)
        else:
            result.setdefault(g, []).append(col)
    return result


__all__ = [
    "FEATURE_GROUPS",
    "assign_column",
    "assign_columns",
    "get_feature_group",
    "group_columns",
]
