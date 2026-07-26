"""Configure DagsHub environment variables for MLflow + DVC.

Run once after cloning::

    uv run python scripts/seed_dagshub.py

Sets up ``.env`` with MLflow tracking URI and DagsHub token.
"""

from __future__ import annotations

import getpass
import os
import webbrowser
from pathlib import Path

_DOTENV = Path(__file__).resolve().parent.parent / ".env"


def main() -> None:
    if _DOTENV.exists():
        overwrite = input(".env exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    print("Connect to DagsHub:\n")
    repo_url = input("DagsHub repo URL (e.g. https://dagshub.com/user/repo): ").strip()
    if not repo_url:
        print("No URL entered — using placeholder values.")
        repo_url = "https://dagshub.com/<user>/<repo>"

    mlflow_uri = f"{repo_url}.mlflow"
    print(f"\nMLflow tracking URI: {mlflow_uri}")
    print("Opening DagsHub settings to generate a token...")
    webbrowser.open("https://dagshub.com/settings/tokens")

    username = input("DagsHub username: ").strip() or "<username>"
    token = getpass.getpass("DagsHub token (hidden): ").strip() or "<token>"

    content = f"""# MLflow tracking (DagsHub managed MLflow)
MLFLOW_TRACKING_URI={mlflow_uri}
MLFLOW_TRACKING_USERNAME={username}
MLFLOW_TRACKING_PASSWORD={token}

# DagsHub token (DVC remote auth)
DAGSHUB_USER_TOKEN={token}
DAGSHUB_REPO_OWNER={username}
DAGSHUB_REPO_NAME={Path(repo_url).name}

# Data location
HC_DATA=./data/raw

# Reproducibility
RANDOM_STATE=42

# Model registry stage to load in API
MODEL_STAGE=Staging
"""
    _DOTENV.write_text(content)
    print(f"\n.env written to {_DOTENV}")
    print("Run: source .env  (or set them in your shell profile)")


if __name__ == "__main__":
    main()