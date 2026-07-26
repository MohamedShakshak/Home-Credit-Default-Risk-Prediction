"""One-shot: download Home Credit data from Kaggle into ``data/raw/``."""

from __future__ import annotations

import subprocess
import sys

_COMPETITION = "home-credit-default-risk"


def main() -> None:
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", _COMPETITION],
            check=True,
        )
        print(f"Downloaded {_COMPETITION}.zip — unzip into data/raw/")
    except FileNotFoundError:
        print(
            "Kaggle CLI not found. Install: pip install kaggle\n"
            "Then place your kaggle.json in ~/.kaggle/ and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()