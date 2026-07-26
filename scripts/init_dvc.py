"""Configure DagsHub as the DVC remote and MLflow tracking server.

Usage::

    uv run python scripts/init_dvc.py

This script will prompt for DagsHub credentials and set up:

- DVC remote (``dagshub``)
- MLflow tracking URI in ``.env``
- ``.dvc/config`` with authentication
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DVC_CONFIG = _REPO_ROOT / ".dvc" / "config"
_DOTENV = _REPO_ROOT / ".env"


def _run_dvc(*args: str) -> None:
    subprocess.run(["dvc", *args], check=True, cwd=_REPO_ROOT)


def main() -> None:
    print("DVC + DagsHub setup\n")

    # ── Gather info ───────────────────────────────────────────────
    dagshub_user = input("DagsHub username: ").strip()
    repo_name = input("DagsHub repo name (default: Home-Credit-Default-Risk-Prediction): ").strip()
    repo_name = repo_name or "Home-Credit-Default-Risk-Prediction"
    dagshub_repo_url = f"https://dagshub.com/{dagshub_user}/{repo_name}"

    token = getpass.getpass("DagsHub token (hidden, generate at https://dagshub.com/settings/tokens): ").strip()
    if not token:
        print("Token is required. Aborting.")
        sys.exit(1)

    # ── DVC remote ────────────────────────────────────────────────
    remote_url = f"{dagshub_repo_url}.dvc"
    print(f"\n→ Setting up DVC remote: {remote_url}")

    try:
        _run_dvc("remote", "add", "-d", "dagshub", remote_url)
    except subprocess.CalledProcessError:
        print("Remote may already exist. Updating...")
        _run_dvc("remote", "modify", "dagshub", "url", remote_url)
        _run_dvc("remote", "default", "dagshub")

    _run_dvc("remote", "modify", "dagshub", "auth", "basic")
    _run_dvc("remote", "modify", "dagshub", "user", dagshub_user)
    _run_dvc("remote", "modify", "dagshub", "password", token)

    print("✓ DVC remote configured.")

    # ── .env file ─────────────────────────────────────────────────
    print(f"\n→ Writing {_DOTENV}")
    mlflow_uri = f"{dagshub_repo_url}.mlflow"

    _DOTENV.write_text(
        f"""# MLflow tracking (DagsHub managed MLflow)
MLFLOW_TRACKING_URI={mlflow_uri}
MLFLOW_TRACKING_USERNAME={dagshub_user}
MLFLOW_TRACKING_PASSWORD={token}

# DagsHub token (DVC remote auth)
DAGSHUB_USER_TOKEN={token}
DAGSHUB_REPO_OWNER={dagshub_user}
DAGSHUB_REPO_NAME={repo_name}

# Data location
HC_DATA=data/raw

# Reproducibility
RANDOM_STATE=42

# Model registry stage to load in API
MODEL_STAGE=Staging
"""
    )
    print("✓ .env written.")

    # ── DVC data tracking (optional) ──────────────────────────────
    raw_dir = _REPO_ROOT / "data" / "raw"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print("\n→ Tracking raw data with DVC...")
        try:
            _run_dvc("add", str(raw_dir))
            print("✓ Raw data tracked. Run `dvc push` to upload.")
        except subprocess.CalledProcessError as exc:
            print(f"! DVC add failed: {exc}")

    print(f"\nDone. Your remote is connected to: {dagshub_repo_url}")
    print("Next steps:")
    print("  1. dvc push      — upload data to DagsHub storage")
    print("  2. dvc repro     — run the full pipeline")
    print("  3. git add -A && git commit -m 'DVC setup'")


if __name__ == "__main__":
    main()
