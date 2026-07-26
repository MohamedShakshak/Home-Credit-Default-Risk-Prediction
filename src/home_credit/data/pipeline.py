"""Orchestrator: load all raw data, run all aggregations, merge into full feature matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.base import BaseEstimator, TransformerMixin

from home_credit.data.application import engineer_application_features
from home_credit.data.bureau import aggregate_bureau, aggregate_bureau_balance
from home_credit.data.loader import load_application
from home_credit.data.previous import (
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_application,
)

if TYPE_CHECKING:
    import pandas as pd


class FullFeaturePipeline(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Load all raw data, engineer all features, merge into one table.

    Sklearn-compatible: ``fit`` is a no-op (all featurization is deterministic
    given raw CSVs); ``transform`` runs the full pipeline and returns
    ``(train_fe, test_fe)``.
    """

    def fit(
        self,
        x: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
    ) -> FullFeaturePipeline:
        return self

    def transform(
        self,
        x: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train, test = load_application()
        return build_full_features(train, test)


def build_full_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all aggregation + engineering, merge into base tables.

    Returns ``(train_fe, test_fe)``, each with ``SK_ID_CURR`` and all
    engineered features. ``TARGET`` is dropped from train_fe.
    """
    _drop = exclude_cols or []

    # ── Application-level engineering ────────────────────────────
    train = engineer_application_features(train)
    test = engineer_application_features(test)

    # ── Auxiliary aggregations ───────────────────────────────────
    bureau_agg = aggregate_bureau()
    bureau_bal_agg = aggregate_bureau_balance()
    prev_app_agg = aggregate_previous_application()
    ins_agg = aggregate_installments()
    pos_agg = aggregate_pos_cash()
    cc_agg = aggregate_credit_card()

    # ── Merge ────────────────────────────────────────────────────
    merges: list[tuple[str, pd.DataFrame]] = [
        ("bureau", bureau_agg),
        ("bureau_bal", bureau_bal_agg),
        ("prev_app", prev_app_agg),
        ("installments", ins_agg),
        ("pos_cash", pos_agg),
        ("credit_card", cc_agg),
    ]
    for _name, agg_df in merges:
        train = train.merge(agg_df, on="SK_ID_CURR", how="left")
        test = test.merge(agg_df, on="SK_ID_CURR", how="left")

    del bureau_agg, bureau_bal_agg, prev_app_agg, ins_agg, pos_agg, cc_agg

    for c in _drop:
        if c in train.columns:
            train = train.drop(columns=[c])
        if c in test.columns:
            test = test.drop(columns=[c])

    return train, test


def train_test_split_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split TARGET off train, align test columns, drop IDs.

    Returns ``(x, y, x_test)``.
    """
    y = train["TARGET"].copy()
    x = train.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    x_test = test.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    x_test = x_test.reindex(columns=x.columns, fill_value=None)
    return x, y, x_test


__all__ = [
    "FullFeaturePipeline",
    "build_full_features",
    "train_test_split_features",
]
