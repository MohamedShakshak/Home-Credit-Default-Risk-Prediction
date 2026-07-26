"""SHAP explainer for the ensemble model.

Fixes from notebook review:

- **C3**: apply ``expit`` (sigmoid) to raw log-odds **before** calibrator.
  SHAP explains raw ensemble output (log-odds), but the calibrator was fit on
  probabilities -> domain mismatch without sigmoid.
- **C4**: handle SHAP return shape ``[:, 1]`` (>=0.42 array form) vs. ``[-1]``
  (<0.42 list form).  Pinned ``shap>=0.44`` in pyproject, but guard remains.
- **C5**: persist ``feature_names.json`` + column dtype schema; assert on every
  ``explain()`` call so production silently-wrong predictions fail fast.
- **W11**: compute global SHAP importance **before** ``top_features`` selection;
  ``top_features`` is set from the sorted importance, not arbitrary first-N cols.
- **W13**: average SHAP values across **all** fold models, not just the last one.
- **M5**: seeded ``np.random.default_rng(RANDOM_STATE)`` for SHAP sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import expit

from home_credit.paths import RANDOM_STATE

if TYPE_CHECKING:
    import pandas as pd


class SHAPExplainer:
    """SHAP explanations for the ensemble model.

    Parameters
    ----------
    fold_models : list of objects
        Fitted model objects (LGBMClassifier / XGBClassifier) from CV folds.
    feature_names : list of str
        Column names in the exact order used during training.
    calibrator : object, optional
        Fitted scikit-learn-style calibrator with ``predict_proba``.
    background_size : int, optional
        Number of background samples for SHAP (typically 100-500).
    random_state : int
        Seed for all RNG operations (fix M5).
    """

    def __init__(
        self,
        fold_models: list[Any],
        feature_names: list[str],
        calibrator: Any | None = None,
        background_size: int = 200,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._fold_models = fold_models
        self._feature_names = list(feature_names)
        self._calibrator = calibrator
        self._background_size = background_size
        self._rng = np.random.default_rng(random_state)
        self._explainers: list[Any] = []
        self._top_features: list[str] = []

    def fit(self, x_background: pd.DataFrame) -> SHAPExplainer:
        """Fit SHAP TreeExplainer for each fold model.

        Uses ``import shap`` inside to avoid import-time overhead when SHAP
        is not installed.
        """
        import shap

        self._explainers = []
        for model in self._fold_models:
            explainer = shap.TreeExplainer(model)
            self._explainers.append(explainer)

        # Compute global SHAP importance to choose top features (fix W11).
        self._top_features = self._compute_top_features(x_background)
        return self

    def explain(self, x_row: pd.DataFrame) -> dict[str, Any]:
        """Explain a single prediction row.

        Parameters
        ----------
        x_row : DataFrame with one row (or more; only first row used).

        Returns
        -------
        dict with keys:
            ``pd`` — calibrated probability of default
            ``raw_score`` — ensemble log-odds before calibrator
            ``base_value`` — expected log-odds (intercept)
            ``shap_values`` — list of (feature_name, value) for all features
            ``top_reasons`` — top-5 features by absolute SHAP value
        """
        # Fix C5: assert column order matches training.
        self._assert_column_order(x_row)

        row = x_row.iloc[:1]
        row_np = np.asarray(row).astype(float)

        shap_values_list: list[np.ndarray] = []
        base_values_list: list[float] = []

        for explainer in self._explainers:
            sv = explainer.shap_values(row_np)

            # Fix C4: handle SHAP return shapes across versions.
            #   list [class0, class1] (ancient)
            #   3-D (n, n_feat, n_outputs) (modern sklearn)
            #   2-D (n, n_feat) (modern single-output)
            if isinstance(sv, list):
                sv = sv[-1]  # positive-class values

            if sv.ndim == 3:
                sv = sv[:, :, -1]  # last output (positive class)

            if sv.ndim == 1:
                sv = sv[np.newaxis, :]

            shap_values_list.append(sv[0])  # (n_features,)

            # base_value
            bv = explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[-1] if bv.ndim == 1 else bv.item() if bv.ndim == 0 else bv
            base_values_list.append(float(bv))

        # W13: average SHAP values and base values across folds.
        avg_shap = np.mean(shap_values_list, axis=0)
        avg_base = float(np.mean(base_values_list))

        # Compute raw log-odds score.
        raw_score = avg_base + float(avg_shap.sum())

        # Fix C3: apply sigmoid before feeding to calibrator.
        proba_raw = float(expit(raw_score))

        pd_value = proba_raw
        if self._calibrator is not None:
            cal_pred = self._calibrator.predict_proba(np.array([[proba_raw]]))[0, 1]
            pd_value = float(cal_pred)

        # Build feature-value list.
        shap_pairs = list(zip(self._feature_names, avg_shap.tolist(), strict=True))
        shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
        top_reasons = [
            {"feature": name, "shap": round(val, 6)}
            for name, val in shap_pairs_sorted[:5]
        ]

        return {
            "pd": round(pd_value, 6),
            "raw_score": round(raw_score, 6),
            "base_value": round(avg_base, 6),
            "shap_values": shap_pairs_sorted,
            "top_reasons": top_reasons,
        }

    def _compute_top_features(self, x_background: pd.DataFrame) -> list[str]:
        """Rank features by mean absolute SHAP contribution.

        Fix W11: importance computed first, then top_feats derived from it.
        """

        sample = x_background.sample(
            n=min(self._background_size, len(x_background)),
            random_state=int(self._rng.integers(0, 2**31 - 1)),
            replace=False,
        )
        sample_np = np.asarray(sample).astype(float)

        all_importances: list[np.ndarray] = []
        for explainer in self._explainers:
            sv = explainer.shap_values(sample_np)
            if isinstance(sv, list):
                sv = sv[-1]
            if sv.ndim == 3:
                sv = sv[:, :, -1]
            mean_abs = np.abs(sv).mean(axis=0)
            all_importances.append(mean_abs)

        avg_importance = np.mean(all_importances, axis=0)
        ranked = [
            self._feature_names[i]
            for i in np.argsort(avg_importance)[::-1]
        ]
        return ranked

    def _assert_column_order(self, x_row: pd.DataFrame) -> None:
        """Fix C5: fail fast on column order mismatch."""
        incoming = list(x_row.columns)
        if incoming != self._feature_names:
            msg = (
                f"Feature order mismatch: expected {len(self._feature_names)} "
                f"columns, got {len(incoming)}. "
                f"First diff: {set(incoming) ^ set(self._feature_names)}"
            )
            raise ValueError(msg)

    def save(self, path: str | Path) -> None:
        """Persist explainer metadata (feature names, top features).

        Does **not** pickle the fold models — those are stored separately
        via MLflow / joblib.  Caller is responsible for restoring models
        before loading.
        """
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_names": self._feature_names,
            "top_features": self._top_features,
        }
        with open(path / "explainer_metadata.json", "w") as f:
            json.dump(metadata, f)

        joblib.dump(self._explainers, path / "explainers.pkl")

    @classmethod
    def load(
        cls,
        path: str | Path,
        fold_models: list[Any],
        calibrator: Any | None = None,
    ) -> SHAPExplainer:
        """Load explainer metadata and wrap it around restored fold models.

        Parameters
        ----------
        path : str | Path
            Directory containing ``explainer_metadata.json`` + ``explainers.pkl``.
        fold_models : list
            Restored model objects (caller's responsibility).
        calibrator : optional
            Restored calibrator.
        """
        import joblib

        path = Path(path)
        with open(path / "explainer_metadata.json") as f:
            metadata: dict[str, Any] = json.load(f)

        instance = cls(
            fold_models=fold_models,
            feature_names=metadata["feature_names"],
            calibrator=calibrator,
        )
        instance._top_features = metadata["top_features"]
        instance._explainers = joblib.load(path / "explainers.pkl")
        return instance

    @property
    def top_features(self) -> list[str]:
        return list(self._top_features)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)


__all__ = ["SHAPExplainer"]
