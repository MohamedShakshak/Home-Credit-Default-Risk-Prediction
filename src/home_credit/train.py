"""Hydra training entrypoint.

Usage::

    python -m home_credit.train data=raw model=blend train=default

Logs to MLflow, registers model to Staging.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np

from home_credit.data.application import engineer_application_features
from home_credit.data.loader import load_application
from home_credit.evaluate.metrics import auc, brier, confusion, expected_loss, ks_statistic
from home_credit.features.encoders import CategoricalEncoder
from home_credit.features.selection import select_features
from home_credit.models.blender import blend_predictions, optimize_blend_weights
from home_credit.models.calibrator import calibrate_with_split
from home_credit.models.lgb_trainer import train_lgb_cv
from home_credit.models.xgb_trainer import train_xgb_cv
from home_credit.registry.mlflow_client import (
    end_run,
    log_dict,
    log_hydra_config,
    log_metrics,
    register_model,
    start_run,
)
from home_credit.registry.pyfunc_ensemble import EnsemblePyFunc, save_ensemble_artifact

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _random_state = cfg.get("random_state", 42)

    # ── Load & featurize ──────────────────────────────────────────
    train, test = load_application()
    train_fe = engineer_application_features(train)
    engineer_application_features(test)

    # Encode categoricals.
    encoder = CategoricalEncoder()
    x_full = encoder.fit_transform(train_fe.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore"))
    y = train_fe["TARGET"].copy()

    # ── Feature selection inside outer CV (each fold selects) ─────
    selected = select_features(
        x_full,
        y,
        method="mutual_info",
        k=cfg.model.get("n_features", 200),
        random_state=_random_state,
    )
    x_sel = x_full[selected]

    # ── LGB CV ────────────────────────────────────────────────────
    _lgb_params = dict(cfg.model.lgb)
    lgb_models, lgb_oof, lgb_fold_aucs = train_lgb_cv(
        x_sel,
        y,
        n_splits=cfg.data.processed.get("n_splits", 5),
        params=_lgb_params,
        random_state=_random_state,
    )
    lgb_mean_auc = float(np.mean(lgb_fold_aucs))

    # ── XGB CV ────────────────────────────────────────────────────
    _xgb_params = dict(cfg.model.xgb)
    xgb_models, xgb_oof, xgb_fold_aucs = train_xgb_cv(
        x_sel,
        y,
        n_splits=cfg.data.processed.get("n_splits", 5),
        params=_xgb_params,
        random_state=_random_state,
    )
    xgb_mean_auc = float(np.mean(xgb_fold_aucs))

    # ── Blend weights (nested CV — fix W6) ────────────────────────
    lgb_weight, _blend_inner_auc = optimize_blend_weights(
        lgb_oof,
        xgb_oof,
        y,
        inner_cv=cfg.model.blend.get("inner_cv", 3),
        random_state=_random_state,
    )
    blended_oof = blend_predictions(lgb_oof, xgb_oof, lgb_weight)
    y_arr = np.asarray(y)
    blend_auc = auc(y_arr, blended_oof)

    # ── Calibration (held-out split — fix C2) ─────────────────────
    cals, cal_eval_preds, cal_brier, best_method = calibrate_with_split(
        blended_oof,
        y,
        test_size=cfg.data.processed.cal_split.get("test_size", 0.5),
        methods=["sigmoid", "isotonic"],
        random_state=_random_state,
    )
    best_cal = cals[["sigmoid", "isotonic"].index(best_method)]

    # ── Final metrics ─────────────────────────────────────────────
    final_auc = auc(y_arr, cal_eval_preds)
    final_brier = brier(y_arr, cal_eval_preds)
    final_ks = ks_statistic(y_arr, cal_eval_preds)
    conf = confusion(
        y_arr, cal_eval_preds, threshold=cfg.model.blend.threshold.get("default", 0.15)
    )
    exp_loss = expected_loss(
        y_arr,
        cal_eval_preds,
        threshold=cfg.model.blend.threshold.get("default", 0.15),
        fp_cost=cfg.model.blend.threshold.get("fp_cost", 1.0),
        fn_cost=cfg.model.blend.threshold.get("fn_cost", 5.0),
    )

    metrics = {
        "auc_lgb_cv_mean": lgb_mean_auc,
        "auc_xgb_cv_mean": xgb_mean_auc,
        "auc_blend_oof": blend_auc,
        "auc_calibrated": final_auc,
        "brier_calibrated": final_brier,
        "ks_calibrated": final_ks,
        "expected_loss": exp_loss,
        "lgb_weight": lgb_weight,
    }
    for i, auc_val in enumerate(lgb_fold_aucs):
        metrics[f"auc_lgb_fold_{i}"] = auc_val
    for i, auc_val in enumerate(xgb_fold_aucs):
        metrics[f"auc_xgb_fold_{i}"] = auc_val

    # ── Log to MLflow ─────────────────────────────────────────────
    start_run(
        run_name=cfg.get("run_name", "train"),
        tags={"git_hash": _git_hash(), "model": "lgb+xgb+blend"},
    )
    log_hydra_config(cfg)
    log_metrics(metrics)
    log_dict("confusion_matrix", conf)
    log_dict("calibrator_info", {"method": best_method, "brier_eval": cal_brier})

    # ── Save artifact + register ──────────────────────────────────
    artifact_dir = Path("model_artifact")
    save_ensemble_artifact(
        artifact_dir,
        fold_models=lgb_models + xgb_models,
        feature_names=list(x_sel.columns),
        lgb_weight=lgb_weight,
        calibrator=best_cal,
        encoder=encoder,
        threshold=cfg.model.blend.threshold.get("default", 0.15),
    )

    model = EnsemblePyFunc()
    register_model(
        model,
        model_name=cfg.model_registry_name,
        stage="Staging",
    )

    print(f"Training complete. AUC: {final_auc:.4f}, Brier: {final_brier:.4f}, KS: {final_ks:.4f}")
    end_run()


def _git_hash() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
