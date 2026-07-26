"""Configuration utilities — Hydra/OmegaConf helpers.

Light-weight config loading for tests and CLI helpers. The production
training entrypoint uses `hydra.main`; this module emulates the same
composition for cases where full Hydra initialization is undesirable
(unit tests, CLI dry-run).
"""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load top-level Hydra config from `configs/config.yaml`.

    Resolves the `defaults:` list — each entry is `{<group>: <name>}` —
    loading `configs/<group>/<name>.yaml` and merging under `cfg[<group>]`.
    The `_self_` marker (if present) is a no-op: the top-level keys in
    `config.yaml` already constitute the "self" portion.
    """
    from home_credit.paths import CONFIG_DIR

    cfg = cast("DictConfig", OmegaConf.load(CONFIG_DIR / "config.yaml"))

    defaults_raw: Any = cfg.pop("defaults", []) if "defaults" in cfg else []
    defaults = cast("ListConfig", defaults_raw)
    processed: set[str] = set()
    for entry in defaults:
        if not hasattr(entry, "items"):
            continue
        for group, name in entry.items():
            if group == "_self_" or name is None:
                continue
            key = f"{group}/{name}"
            if key in processed:
                continue
            processed.add(key)
            sub_path = CONFIG_DIR / group / f"{name}.yaml"
            if sub_path.exists():
                sub_cfg = OmegaConf.load(sub_path)
                cfg.merge_with({group: sub_cfg})

    if overrides:
        cfg.merge_with(OmegaConf.from_dotlist(overrides))

    return cfg


def validate_config(cfg: DictConfig) -> list[str]:
    """Return a list of validation errors. Empty list = valid config."""
    errors: list[str] = []
    for key in ("data", "model", "train", "api"):
        if key not in cfg:
            errors.append(f"missing top-level key: {key}")
    if "random_state" not in cfg:
        errors.append("missing random_state")
    if int(cfg.get("random_state", 0)) <= 0:
        errors.append("random_state must be positive")
    return errors


__all__ = ["load_config", "validate_config"]
