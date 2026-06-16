from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


A2_SELECTED_COLUMNS = [
    "pmts_dpd_1073P",
    "pmts_dpd_303P",
    "pmts_overdue_1140A",
    "pmts_overdue_1152A",
    "pmts_month_158T",
    "pmts_month_706T",
    "pmts_year_1139T",
    "pmts_year_507T",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile raw credit_bureau_a_2 selected columns before aggregation.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_lab/a2_raw_profile.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.data_dir / "parquet_files" / args.split).glob(f"{args.split}_credit_bureau_a_2_*.parquet"))
    if not paths:
        raise FileNotFoundError("No credit_bureau_a_2 parquet files found.")

    rows = []
    for path in paths:
        scan = pl.scan_parquet(str(path))
        schema = scan.collect_schema()
        available = [col for col in A2_SELECTED_COLUMNS if col in schema]
        exprs = [pl.len().alias("__rows")]
        for col in available:
            base = pl.col(col).cast(pl.Float64, strict=False)
            exprs.extend(
                [
                    base.null_count().alias(f"{col}__nulls"),
                    base.min().alias(f"{col}__min"),
                    base.max().alias(f"{col}__max"),
                    base.mean().alias(f"{col}__mean"),
                ]
            )
        stats = scan.select(exprs).collect(engine="streaming").to_dicts()[0]
        total_rows = int(stats["__rows"])
        for col in available:
            rows.append(
                {
                    "file": path.name,
                    "feature": col,
                    "rows": total_rows,
                    "missing_rate": float(stats[f"{col}__nulls"] / total_rows) if total_rows else 0.0,
                    "min": stats[f"{col}__min"],
                    "max": stats[f"{col}__max"],
                    "mean": stats[f"{col}__mean"],
                }
            )
        print(json.dumps({"file": path.name, "rows": total_rows, "features": len(available)}))

    pl.DataFrame(rows).write_csv(args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
