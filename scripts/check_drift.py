"""Drift check script: compare reference features against new data.

Usage::

    uv run python scripts/check_drift.py \\
        --reference data/interim/train_fe.parquet \\
        --current data/interim/current_fe.parquet \\
        --output drift_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from home_credit.evaluate.drift import drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect feature drift between reference and current data")
    parser.add_argument("--reference", required=True, help="Reference parquet/CSV (training set)")
    parser.add_argument("--current", required=True, help="Current parquet/CSV (new inference data)")
    parser.add_argument("--output", default="drift_report.json", help="Output JSON path")
    parser.add_argument("--psi-threshold", type=float, default=0.1, help="PSI threshold for flagging drift (default: 0.1)")
    parser.add_argument("--ks-threshold", type=float, default=0.2, help="KS threshold for flagging drift (default: 0.2)")
    parser.add_argument("--fmt", choices=["auto", "parquet", "csv"], default="auto", help="File format")
    args = parser.parse_args()

    ref_df = _load(args.reference, args.fmt)
    cur_df = _load(args.current, args.fmt)

    common = set(ref_df.columns) & set(cur_df.columns)
    if not common:
        print("No common columns between reference and current datasets.")
        return

    # Drop non-numeric columns from drift check.
    numeric_cols = [
        c for c in common
        if pd.api.types.is_numeric_dtype(ref_df[c]) and pd.api.types.is_numeric_dtype(cur_df[c])
    ]
    if not numeric_cols:
        print("No common numeric columns for drift analysis.")
        return

    ref_dict = {c: ref_df[c].values for c in numeric_cols}
    cur_dict = {c: cur_df[c].values for c in numeric_cols}

    result = drift_report(ref_dict, cur_dict)

    # Flag drifted features.
    drifted = [
        feat for feat, m in result.items()
        if m["psi"] > args.psi_threshold or m["ks"] > args.ks_threshold
    ]

    # Round floats for readability.
    clean: dict = {}
    for feat, m in result.items():
        clean[feat] = {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
        clean[feat]["drifted"] = feat in drifted

    report = {
        "n_features_checked": len(clean),
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "features": clean,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Drift report written to {args.output}")
    print(f"  Features checked: {len(clean)}")
    print(f"  Drifted features: {len(drifted)}")
    if drifted:
        print(f"  Top drifted: {drifted[:5]}")


def _load(path: str, fmt: str) -> pd.DataFrame:
    p = Path(path)
    if fmt == "parquet" or (fmt == "auto" and p.suffix in (".parquet", ".pq")):
        return pd.read_parquet(p)
    return pd.read_csv(p)


if __name__ == "__main__":
    main()
