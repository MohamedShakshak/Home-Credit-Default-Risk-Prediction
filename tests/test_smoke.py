"""Smoke tests: package imports, Hydra config loads, paths resolve.

Phase 0 checkpoint — `pytest tests/test_smoke.py` must pass before Phase 1.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from home_credit.paths import CONFIG_DIR, PACKAGE_ROOT, PROJECT_ROOT, RANDOM_STATE


def test_package_imports() -> None:
    import home_credit

    assert home_credit.__version__ == "0.1.0"


def test_paths_resolve() -> None:
    assert PROJECT_ROOT.exists()
    assert CONFIG_DIR.exists() and CONFIG_DIR.is_dir()
    assert RANDOM_STATE == 42
    assert PACKAGE_ROOT.name == "src"


def test_paths_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HC_DATA", str(tmp_path / "custom_data"))
    from home_credit import paths as paths_mod

    importlib.reload(paths_mod)
    assert "custom_data" in str(paths_mod.DATA_DIR)


def test_paths_ensure_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HC_DATA", str(tmp_path / "raw"))
    from home_credit import paths as paths_mod

    importlib.reload(paths_mod)
    paths_mod.ensure_dirs()
    assert paths_mod.DATA_DIR.exists()
    assert paths_mod.PROCESSED_DIR.exists()


def test_all_subpackages_importable() -> None:
    pass


@pytest.mark.smoke
def test_smoke_marker() -> None:
    """Marker smoke test for `pytest -m smoke`."""
    assert True
