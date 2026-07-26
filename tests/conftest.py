"""Pytest fixtures shared across tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable when package not yet installed (e.g. CI before uv sync).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Path to the Hydra configs directory."""
    return _CONFIGS


@pytest.fixture
def random_state() -> int:
    """Global random seed — every test references this."""
    return 42
