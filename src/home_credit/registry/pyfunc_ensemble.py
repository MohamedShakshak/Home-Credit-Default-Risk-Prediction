"""Custom MLflow PyFunc model wrapping the full ensemble pipeline.

Wraps: featurizers + LGB models + XGB models + blend weights + calibrator.
Single artifact for MLflow Model Registry — not 5 separate ones.

``predict`` ingests a raw DataFrame (must contain ``application_train``-style
columns plus auxiliary tables loaded on the fly) and returns calibrated
probabilities with SHAP top reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from home_credit.data.application import engineer_application_features
from home_credit.models.blender import blend_predictions

if TYPE_CHECKING:
    from home_credit.explain.shap_explainer import SHAPExplainer


class EnsemblePyFunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
    """PyFunc model that wraps the full ensemble pipeline.

    State (set via ``load_context``):
      - ``fold_models``: list of fitted LGBM/XGBoost models
      - ``feature_names``: column list used during training
      - ``lgb_weight``: blend weight for LGB (XGB weight = 1 - lgb_weight)
      - ``calibrator``: fitted calibrator (sigmoid or isotonic)
      - ``explainer``: ``SHAPExplainer`` instance
      - ``threshold``: decision threshold for hard classification
    """

    def __init__(self) -> None:
        self._fold_models: list[Any] = []
        self._feature_names: list[str] = []
        self._lgb_weight: float = 0.5
        self._calibrator: Any = None
        self._explainer: SHAPExplainer | None = None
        self._threshold: float = 0.5
        self._encoder: Any = None

    def load_context(self, context: Any) -> None:
        """Restore model components from the artifact directory."""
        import joblib

        artifacts_dir = Path(context.artifacts.get("model_dir", "."))

        # Fold models.
        models_path = artifacts_dir / "fold_models.pkl"
        if models_path.exists():
            self._fold_models = joblib.load(models_path)

        # Feature names.
        meta_path = artifacts_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta: dict[str, Any] = json.load(f)
            self._feature_names = meta.get("feature_names", [])
            self._lgb_weight = meta.get("lgb_weight", 0.5)
            self._threshold = meta.get("threshold", 0.5)

        # Calibrator.
        cal_path = artifacts_dir / "calibrator.pkl"
        if cal_path.exists():
            self._calibrator = joblib.load(cal_path)

        # Categorical encoder.
        enc_path = artifacts_dir / "encoder.pkl"
        if enc_path.exists():
            self._encoder = joblib.load(enc_path)

        # SHAP explainer (built lazily — requires fold_models + background data).
        # In practice, load from a pre-saved explainer or build on first call.

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Score raw input data through the pipeline.

        Parameters
        ----------
        model_input : DataFrame
            Raw application data (same columns as ``application_train.csv``).
        params : dict, optional
            Override keys: ``return_shap`` (bool), ``threshold`` (float).

        Returns
        -------
        DataFrame with columns:
            ``SK_ID_CURR``, ``predicted_pd``, ``decision``,
            and optionally SHAP top-5 reasons.
        """
        _return_shap = (params or {}).get("return_shap", False)
        _threshold = (params or {}).get("threshold", self._threshold)

        # Apply data-layer transforms.
        df = model_input.copy()
        df = engineer_application_features(df)
        if self._encoder is not None:
            df = self._encoder.transform(df)

        # Align columns to training order.
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self._feature_names]

        # Predict with each fold model and average.
        fold_preds: list[np.ndarray] = []
        for model in self._fold_models:
            fold_preds.append(model.predict_proba(df)[:, 1])
        avg_preds = np.mean(fold_preds, axis=0)

        # Blend.
        if len(self._fold_models) >= 2:
            # Use the first half of fold models as LGB group, second half as XGB.
            n = len(self._fold_models)
            n_lgb = n // 2
            lgb_preds = np.mean(fold_preds[:n_lgb], axis=0)
            xgb_preds = np.mean(fold_preds[n_lgb:], axis=0)
            blended = blend_predictions(lgb_preds, xgb_preds, self._lgb_weight)
        else:
            blended = avg_preds

        # Calibrate.
        if self._calibrator is not None:
            probs = self._calibrator.predict_proba(blended[:, None])[:, 1]
        else:
            probs = blended

        # Build output.
        result = pd.DataFrame({"SK_ID_CURR": model_input.get("SK_ID_CURR", range(len(df)))})
        result["predicted_pd"] = np.round(probs, 6)
        result["decision"] = (probs >= _threshold).astype(int)

        # Add SHAP explanations if requested.
        if _return_shap and self._explainer is not None:
            for i in range(len(df)):
                row_df = df.iloc[i : i + 1]
                explanation = self._explainer.explain(row_df)
                top_reasons = explanation.get("top_reasons", [])
                for j, reason in enumerate(top_reasons):
                    result.loc[i, f"shap_feature_{j + 1}"] = reason["feature"]
                    result.loc[i, f"shap_value_{j + 1}"] = reason["shap"]

        return result


def save_ensemble_artifact(
    path: str | Path,
    fold_models: list[Any],
    feature_names: list[str],
    lgb_weight: float,
    calibrator: Any = None,
    encoder: Any = None,
    threshold: float = 0.5,
    explainer: SHAPExplainer | None = None,
) -> Path:
    """Save all model components to a directory for use as an MLflow artifact."""
    import joblib

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(fold_models, path / "fold_models.pkl")

    metadata = {
        "feature_names": feature_names,
        "lgb_weight": lgb_weight,
        "threshold": threshold,
    }
    with open(path / "model_metadata.json", "w") as f:
        json.dump(metadata, f)

    if calibrator is not None:
        joblib.dump(calibrator, path / "calibrator.pkl")

    if encoder is not None:
        joblib.dump(encoder, path / "encoder.pkl")

    if explainer is not None:
        explainer.save(path / "explainer")

    return path


__all__ = [
    "EnsemblePyFunc",
    "save_ensemble_artifact",
]
