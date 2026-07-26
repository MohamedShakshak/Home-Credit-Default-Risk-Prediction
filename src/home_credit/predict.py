"""CLI: score a CSV file from the MLflow registry.

Usage::

    echo 'SK_ID_CURR,AMT_INCOME_TOTAL,AMT_CREDIT,...' | uv run python -m home_credit.predict
    uv run python -m home_credit.predict --input data/raw/application_test.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys

import mlflow.pyfunc
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CSV from MLflow registry")
    parser.add_argument("--input", "-i", type=str, help="Input CSV path (default: stdin)")
    parser.add_argument(
        "--output", "-o", type=str, default="predictions.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default="models:/home_credit_default/Staging",
        help="MLflow model URI",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Decision threshold")
    parser.add_argument("--return-shap", action="store_true", help="Include SHAP top reasons")
    args = parser.parse_args()

    df = pd.read_csv(args.input) if args.input else pd.read_csv(sys.stdin)

    model = mlflow.pyfunc.load_model(args.model_uri)

    params = {
        "threshold": args.threshold,
        "return_shap": args.return_shap,
    }
    result = model.predict(df, params=params)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} rows -> {args.output}")


if __name__ == "__main__":
    main()
