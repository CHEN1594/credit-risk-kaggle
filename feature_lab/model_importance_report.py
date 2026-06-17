from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightGBM gain/split importance and optionally merge IV/PSI.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_lab/model_importance.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.artifact_dir / "model.joblib")
    feature_names = list(model.booster_.feature_name())
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    out["gain_rank"] = out["importance_gain"].rank(method="min", ascending=False).astype(int)
    out["split_rank"] = out["importance_split"].rank(method="min", ascending=False).astype(int)

    if args.feature_report is not None and args.feature_report.exists():
        report = pd.read_csv(args.feature_report)
        keep_cols = [
            col
            for col in ["feature", "missing_rate", "nunique", "iv", "psi", "univariate_auc", "bin_monotonic_score"]
            if col in report.columns
        ]
        out = out.merge(report[keep_cols], on="feature", how="left")

    out = out.sort_values(["importance_gain", "importance_split"], ascending=[False, False])
    out.to_csv(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
