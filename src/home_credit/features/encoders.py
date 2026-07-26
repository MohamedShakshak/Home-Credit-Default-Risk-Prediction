"""Encoders: ordinal encoding (object cols) + target encoding with fold isolation.

Fixes notebook M6: docstring now matches implementation (Bayesian smoothing).
Fixes notebook W1: encoding fit on train-fold only via ``fit_transform`` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

if TYPE_CHECKING:
    import pandas as pd


class CategoricalEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Ordinal-encode all ``object``-dtype columns.

    Unknown categories encountered at transform time are encoded as ``-1``.
    Missing values (``NaN``) are encoded as ``-2``.
    """

    def __init__(self) -> None:
        self._encoder = SkOrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        self._cat_cols: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> CategoricalEncoder:
        self._cat_cols = x.select_dtypes(include="object").columns.tolist()
        if self._cat_cols:
            self._encoder.fit(x[self._cat_cols])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        if self._cat_cols:
            x[self._cat_cols] = self._encoder.transform(x[self._cat_cols]).astype("int32")
        return x


class TargetEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Bayesian target encoder with fold isolation.

    Encodes each category as a weighted blend of the category-level target mean
    and the global target mean (smoothing):

        encoded = (n * mean + m * global_mean) / (n + m)

    Parameters
    ----------
    m : float, default=10.0
        Smoothing strength. Higher values pull estimates harder toward the
        global mean (stronger regularisation).
    handle_unknown : str, default="global_mean"
        Strategy for unseen categories at transform time.
    """

    def __init__(self, m: float = 10.0, handle_unknown: str = "global_mean") -> None:
        self.m = m
        self.handle_unknown = handle_unknown
        self._mapping: dict[str, dict[Any, float]] = {}
        self._global_mean: float = 0.0
        self._cat_cols: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> TargetEncoder:
        import pandas as pd

        self._cat_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
        self._global_mean = float(y.mean())

        for col in self._cat_cols:
            stats = (
                pd.DataFrame({"target": y, "col": x[col]})
                .groupby("col")
                .agg(
                    count=("target", "count"),
                    mean=("target", "mean"),
                )
            )
            stats["encoded"] = (stats["count"] * stats["mean"] + self.m * self._global_mean) / (
                stats["count"] + self.m
            )
            self._mapping[col] = stats["encoded"].to_dict()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for col in self._cat_cols:
            if col not in x.columns:
                continue
            encoded = x[col].map(self._mapping.get(col, {}))
            if self.handle_unknown == "global_mean":
                encoded = encoded.fillna(self._global_mean)
            elif self.handle_unknown == "zero":
                encoded = encoded.fillna(0.0)
            else:
                encoded = encoded.fillna(self._global_mean)
            x[col] = encoded.astype("float32")
        return x

    @property
    def category_counts(self) -> dict[str, int]:
        return {col: len(m) for col, m in self._mapping.items()}


__all__ = ["CategoricalEncoder", "TargetEncoder"]
