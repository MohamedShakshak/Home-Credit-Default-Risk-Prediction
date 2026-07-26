"""Per-fold feature selection (fix W1: no target-informed selection on full dataset).

All selection functions take ``(x_tr, y_tr)`` only — never the full dataset.
This ensures no data leakage across the train/validation boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.feature_selection import mutual_info_classif

if TYPE_CHECKING:
    import pandas as pd


def select_by_mutual_info(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    k: int | float | None = None,
    percentile: float | None = None,
    random_state: int = 42,
    min_features: int = 10,
) -> list[str]:
    """Select top-``k`` features by mutual information with the target.

    ``k`` may be an absolute count or a fraction of total features (0.0-1.0).
    If both ``k`` and ``percentile`` are ``None``, all features are returned.
    At least ``min_features`` are always returned.
    """
    if k is None and percentile is None:
        return list(x_tr.columns)

    n = len(x_tr.columns)
    if percentile is not None:
        k = max(int(n * percentile / 100), min_features)
    elif isinstance(k, float):
        k = max(int(n * k), min_features)
    elif k is None:
        k = n
    else:
        k = max(int(k), min_features)

    k = min(k, n)

    mi = mutual_info_classif(x_tr.fillna(0), y_tr, random_state=random_state)
    ranked = [col for _, col in sorted(zip(mi, x_tr.columns, strict=True), reverse=True)]
    return ranked[:k]


def select_by_variance(
    x_tr: pd.DataFrame,
    *,
    threshold: float = 0.0,
    percentile: float | None = None,
) -> list[str]:
    """Select features whose variance exceeds ``threshold``.

    When ``percentile`` is set, the threshold is taken as the given percentile
    of all variances.
    """
    variances = x_tr.var(numeric_only=True).fillna(0)
    actual_threshold = threshold
    if percentile is not None:
        actual_threshold = float(np.percentile(variances, percentile))
    selected = variances[variances > actual_threshold].index.tolist()
    return selected


def select_constant_and_duplicate(
    x_tr: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return (constant_cols, duplicate_cols) — columns to drop."""
    constant: list[str] = []
    duplicate: list[str] = []
    seen: dict[str, Any] = {}
    for col in x_tr.columns:
        if x_tr[col].nunique(dropna=True) <= 1:
            constant.append(col)
            continue
        if x_tr[col].dtype.kind in ("i", "f", "b"):
            try:
                hashed = x_tr[col].fillna(-999).to_numpy().tobytes()
            except (ValueError, TypeError, AttributeError):
                continue
            for _existing_col, existing_hash in seen.items():
                if isinstance(existing_hash, bytes) and hashed == existing_hash:
                    duplicate.append(col)
                    break
            else:
                seen[col] = hashed
    return constant, duplicate


def select_features(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    *,
    method: Literal["mutual_info", "variance"] | None = None,
    k: int | float | None = 200,
    min_features: int = 10,
    random_state: int = 42,
) -> list[str]:
    """Run the default per-fold selection pipeline.

    1. Drop constant and duplicate columns.
    2. Drop zero-variance columns.
    3. If ``method="mutual_info"``, select top-``k`` by mutual information
       (only if more than ``k`` columns remain).

    .. note::

        All decisions are based exclusively on ``(x_tr, y_tr)`` — the
        training fold.  The ``y_tr`` argument is never None when
        ``method="mutual_info"``.
    """
    constant, duplicate = select_constant_and_duplicate(x_tr)
    to_drop = set(constant + duplicate)

    remaining = [c for c in x_tr.columns if c not in to_drop]
    if not remaining:
        return list(x_tr.columns)

    x_clean = x_tr[remaining]

    # Variance filter.
    high_var = select_by_variance(x_clean, threshold=1e-8)
    if not high_var:
        return remaining[:min_features]

    if method == "mutual_info":
        if k is not None and len(high_var) > k:
            return select_by_mutual_info(
                x_tr[high_var],
                y_tr,
                k=k,
                min_features=min_features,
                random_state=random_state,
            )
        return select_by_mutual_info(
            x_tr[high_var],
            y_tr,
            k=None,
            min_features=min_features,
            random_state=random_state,
        )

    return high_var


__all__ = [
    "select_by_mutual_info",
    "select_by_variance",
    "select_constant_and_duplicate",
    "select_features",
]
