from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


TABLE_GROUPS: dict[str, list[str]] = {
    "drop_credit_bureau_a_1": ["credit_bureau_a_1__"],
    "drop_credit_bureau_a_2": ["credit_bureau_a_2__"],
    "drop_applprev_1": ["applprev_1__"],
    "drop_tax_all": ["tax_registry_a_1__", "tax_registry_b_1__", "tax_registry_c_1__"],
    "drop_person_all": ["person_1__", "person_2__"],
    "drop_deposit_debitcard": ["deposit_1__", "debitcard_1__"],
    "drop_small_external": ["tax_registry_a_1__", "tax_registry_b_1__", "tax_registry_c_1__", "deposit_1__", "debitcard_1__"],
}


AGG_GROUPS: dict[str, list[str]] = {
    "drop_sum": ["sum"],
    "drop_std": ["std"],
    "drop_median": ["median"],
    "drop_first_last": ["first", "last"],
    "drop_sum_std": ["sum", "std"],
    "drop_count": ["count"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablation variants and strict CV.")
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments_v13_ablation"))
    parser.add_argument("--suite", choices=["table", "agg"], default="table")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean_gini": payload.get("mean_gini"),
        "min_gini": payload.get("min_gini"),
        "std_gini": payload.get("std_gini"),
        "last20_gini": payload.get("last20_gini"),
        "mean_stability": payload.get("mean_stability"),
        "min_stability": payload.get("min_stability"),
        "folds_built": payload.get("validation", {}).get("folds_built"),
    }


def base_feature_columns(base_run_dir: Path) -> list[str]:
    metadata = json.loads((base_run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    schema = pq.ParquetFile(train_path).schema_arrow
    return [name for name in schema.names if name not in {"case_id", "target", "WEEK_NUM"}]


def write_drop_list(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature"])
        writer.writeheader()
        for col in columns:
            writer.writerow({"feature": col})


def main() -> None:
    args = parse_args()
    groups = TABLE_GROUPS if args.suite == "table" else AGG_GROUPS
    selected = args.only or list(groups)
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_cols = base_feature_columns(args.base_run_dir)
    rows = []

    for name in selected:
        if name not in groups:
            raise ValueError(f"Unknown ablation {name!r}. Available: {sorted(groups)}")
        run_dir = args.output_root / name
        cv_path = args.output_root / f"{name}_cv.json"
        if args.suite == "table":
            drop_cols = [col for col in base_cols if any(col.startswith(prefix) for prefix in groups[name])]
        else:
            drop_cols = [col for col in base_cols if any(col.endswith(f"__{agg}") for agg in groups[name])]

        drop_list = args.output_root / "drop_lists" / f"{name}.csv"
        write_drop_list(drop_list, drop_cols)
        if not run_dir.exists():
            run(
                [
                    sys.executable,
                    "scripts/make_feature_variant.py",
                    "--base-run-dir",
                    str(args.base_run_dir),
                    "--output-run-dir",
                    str(run_dir),
                    "--drop-list",
                    str(drop_list),
                    "--max-rss-gb",
                    str(args.max_rss_gb),
                    "--min-available-gb",
                    str(args.min_available_gb),
                ]
            )

        run(
            [
                sys.executable,
                "scripts/cv_features.py",
                "--run-dir",
                str(run_dir),
                "--output-path",
                str(cv_path),
                "--model",
                "lightgbm",
                "--n-estimators",
                str(args.n_estimators),
                "--early-stopping-rounds",
                str(args.early_stopping_rounds),
                "--max-rss-gb",
                str(args.max_rss_gb),
                "--min-available-gb",
                str(args.min_available_gb),
            ]
        )
        rows.append({"ablation": name, **read_summary(cv_path)})

    summary_path = args.output_root / f"{args.suite}_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(summary_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
