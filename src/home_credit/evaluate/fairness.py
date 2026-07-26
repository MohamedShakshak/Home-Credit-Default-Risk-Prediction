"""Fairness audit: demographic parity, index-aligned joins (fix W12).

Fixes notebook W12: uses merge on ``SK_ID_CURR`` instead of
``.values`` assignment that can silently misalign demographics
when rows are dropped during feature processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def fairness_report(
    predictions: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    pred_col: str = "predicted_pd",
    target_col: str = "TARGET",
    protected_attrs: list[str] | None = None,
) -> dict[str, Any]:
    """Compute fairness metrics per protected attribute.

    Parameters
    ----------
    predictions : DataFrame
        Must contain ``SK_ID_CURR``, ``pred_col``, and optionally ``target_col``.
    demographics : DataFrame
        Must contain ``SK_ID_CURR`` and the columns in ``protected_attrs``.
    protected_attrs : list of str, optional
        Demographic columns to evaluate (e.g. ``["CODE_GENDER", "AGE_GROUP"]``).

    Returns
    -------
    report : dict
        ``{attr: {"groups": {group_name: {count, mean_pred, mean_target,
        disparity_from_overall}}}, "row_count_mismatch": bool}``
    """
    _attrs = protected_attrs or ["CODE_GENDER"]

    # Merge on SK_ID_CURR — index-aligned join (fix W12).
    merged = predictions.merge(
        demographics[["SK_ID_CURR", *_attrs]],
        on="SK_ID_CURR",
        how="left",
    )

    row_count_mismatch = len(merged) != len(predictions)

    overall_mean_pred = float(merged[pred_col].mean())

    report: dict[str, Any] = {
        "row_count_mismatch": row_count_mismatch,
    }

    for attr in _attrs:
        if attr not in merged.columns:
            continue
        groups: dict[str, Any] = {}
        for name, grp in merged.groupby(attr):
            mean_pred = float(grp[pred_col].mean())
            group_report: dict[str, Any] = {
                "count": len(grp),
                "mean_pred": mean_pred,
                "disparity": mean_pred - overall_mean_pred,
            }
            if target_col in grp.columns:
                group_report["mean_target"] = float(grp[target_col].mean())
            groups[str(name)] = group_report
        report[attr] = {
            "groups": groups,
            "overall_mean_pred": overall_mean_pred,
        }

    return report


def merge_on_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Safely merge two DataFrames on ``id_col``, asserting no row loss.

    Raises ``ValueError`` if the merge would silently change row count.
    """
    result = left.merge(right, on=id_col, how="left")
    if len(result) != len(left):
        raise ValueError(
            f"Merge changed row count: {len(left)} → {len(result)}. "
            "Check for duplicates in the right DataFrame.",
        )
    return result


__all__ = [
    "fairness_report",
    "merge_on_id",
]
