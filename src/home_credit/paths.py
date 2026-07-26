"""Canonical paths for the project.

All paths resolve relative to `DATA_DIR` (env `HC_DATA`, default `./data/raw`).
Never hardcode Kaggle paths — fixes notebooks C1 leak (critical).
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent  # src/
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = Path(os.environ.get("HC_DATA", str(PROJECT_ROOT / "data" / "raw"))).resolve()
INTERIM_DIR: Path = DATA_DIR.parent / "interim"
PROCESSED_DIR: Path = DATA_DIR.parent / "processed"
MODELS_DIR: Path = DATA_DIR.parent / "models"
API_LOG_DIR: Path = PROJECT_ROOT / "api" / "logs"
DAGSHUB_REPO_OWNER: str | None = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME: str = os.environ.get("DAGSHUB_REPO_NAME", "Home-Credit-Default-Risk-Prediction")
RANDOM_STATE: int = int(os.environ.get("RANDOM_STATE", "42"))


def ensure_dirs() -> None:
    """Create runtime directories (idempotent)."""
    for d in (DATA_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, API_LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "API_LOG_DIR",
    "CONFIG_DIR",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_REPO_OWNER",
    "DATA_DIR",
    "INTERIM_DIR",
    "MODELS_DIR",
    "PACKAGE_ROOT",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "ensure_dirs",
]
