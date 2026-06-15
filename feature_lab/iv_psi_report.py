from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple IV and PSI reports for a filtered feature parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_lab/iv_psi_report.csv"))
    parser.add_argument("--target", default="target")
    parser.add_argument("--week", default="WEEK_NUM")
    parser.add_argument("--max-features", type=int, default=300)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def iv_for_series(x: pd.Series, y: pd.Series, bins: int) -> float:
    if pd.api.types.is_bool_dtype(x):
        x = x.astype(float)
    if x.nunique(dropna=True) <= 1:
        return 0.0
    try:
        bucket = pd.qcut(x.rank(method="first"), q=bins, duplicates="drop")
    except ValueError:
        return 0.0
    table = pd.DataFrame({"bucket": bucket, "target": y}).groupby("bucket", observed=True)["target"]
    bad = table.sum().astype(float) + 0.5
    total = table.count().astype(float) + 1.0
    good = total - bad
    bad_dist = bad / bad.sum()
    good_dist = good / good.sum()
    return float(((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum())


def psi_for_series(x: pd.Series, weeks: pd.Series, bins: int) -> float:
    if pd.api.types.is_bool_dtype(x):
        x = x.astype(float)
    valid = x.notna() & weeks.notna()
    if valid.sum() == 0 or x[valid].nunique() <= 1:
        return 0.0
    week_values = np.sort(weeks[valid].unique())
    if len(week_values) < 4:
        return 0.0
    split = week_values[len(week_values) // 2]
    ref = x[valid & (weeks <= split)]
    cur = x[valid & (weeks > split)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_dist = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_dist = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(args.sample_rows, random_state=42)
    y = df[args.target].astype(int)
    weeks = df[args.week]
    candidates = [
        col
        for col in df.columns
        if col not in {"case_id", args.target, args.week}
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 1
    ][: args.max_features]
    rows = []
    for col in candidates:
        x = df[col]
        rows.append(
            {
                "feature": col,
                "iv": iv_for_series(x, y, args.bins),
                "psi": psi_for_series(x, weeks, args.bins),
                "missing_rate": float(x.isna().mean()),
                "nunique": int(x.nunique(dropna=True)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["iv", "psi"], ascending=[False, True])
    out.to_csv(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
