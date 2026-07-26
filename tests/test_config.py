"""Test Hydra/OmegaConf config composition loads and validates."""

from __future__ import annotations

from home_credit.config import load_config, validate_config


def test_load_config_returns_dictconfig() -> None:
    cfg = load_config()
    assert cfg.random_state == 42
    assert cfg.project_name == "home_credit_default"


def test_config_has_required_groups() -> None:
    cfg = load_config()
    for key in ("data", "model", "train", "api"):
        assert key in cfg, f"missing group: {key}"


def test_data_group_has_target_and_id() -> None:
    cfg = load_config()
    assert cfg.data.target_col == "TARGET"
    assert cfg.data.id_col == "SK_ID_CURR"


def test_validate_config_empty() -> None:
    cfg = load_config()
    assert validate_config(cfg) == []


def test_model_blend_has_calibrator() -> None:
    cfg = load_config()
    assert "calibrator" in cfg.model
    assert cfg.model.blend.method == "search"


def test_xgb_handles_nan_natively() -> None:
    """Regression for bug W7 — XGB must not fillna(-999)."""
    cfg = load_config()
    assert cfg.model.xgb.handle_missing == "native"


def test_ensemble_auc_is_dynamic() -> None:
    """Regression for bug W14 — ensemble_auc not hardcoded."""
    cfg = load_config()
    assert cfg.train.ensemble_auc is None


def test_random_state_consistent() -> None:
    cfg = load_config()
    assert cfg.random_state == cfg.train.seed == 42
