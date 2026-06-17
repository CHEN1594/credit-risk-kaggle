from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative feature drop list from health/importance reports.")
    parser.add_argument("--health-dir", type=Path, required=True)
    parser.add_argument("--importance-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-important-gain-to-drop-duplicate", type=float, default=1_000.0)
    parser.add_argument("--high-psi-threshold", type=float, default=2.0)
    parser.add_argument("--max-important-gain-to-drop-high-psi", type=float, default=2_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duplicates = pd.read_csv(args.health_dir / "duplicates.csv")
    profile = pd.read_csv(args.health_dir / "feature_profile.csv")
    importance = pd.read_csv(args.importance_report)
    imp = importance.set_index("feature")

    drop_reasons: dict[str, str] = {}

    for _, row in duplicates.iterrows():
        feature = str(row["feature"])
        duplicate_of = str(row["duplicate_of"])
        gain = float(imp["importance_gain"].get(feature, 0.0)) if feature in imp.index else 0.0
        original_gain = float(imp["importance_gain"].get(duplicate_of, 0.0)) if duplicate_of in imp.index else 0.0
        if gain <= args.max_important_gain_to_drop_duplicate or original_gain >= gain:
            drop_reasons[feature] = f"duplicate_of:{duplicate_of}"

    for _, row in profile.iterrows():
        feature = str(row["feature"])
        if bool(row.get("is_constant", False)):
            drop_reasons.setdefault(feature, "constant_in_profile_sample")

    if {"psi", "importance_gain"}.issubset(importance.columns):
        high_psi = importance[
            (importance["psi"] >= args.high_psi_threshold)
            & (importance["importance_gain"] <= args.max_important_gain_to_drop_high_psi)
        ]
        for _, row in high_psi.iterrows():
            feature = str(row["feature"])
            drop_reasons.setdefault(feature, f"high_psi_low_gain:{float(row['psi']):.4f}")

    rows = [{"feature": feature, "reason": reason} for feature, reason in sorted(drop_reasons.items())]
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(json.dumps({"output": str(args.output), "dropped": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
