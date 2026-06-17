from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.memory import MemoryLimitExceeded, check_memory
from src.preprocess import reduce_mem_usage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a filtered feature run-dir variant by dropping selected columns.")
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--drop-list", type=Path, required=True, help="CSV with a feature column, or a JSON/list text file.")
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    return parser.parse_args()


def read_drop_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        col = "feature" if "feature" in frame.columns else frame.columns[0]
        return [str(value) for value in frame[col].dropna().to_list()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(value) for value in payload]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    args.output_run_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.base_run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    check_memory("before read base train", args.max_rss_gb, args.min_available_gb)
    df = pd.read_parquet(train_path)
    check_memory("after read base train", args.max_rss_gb, args.min_available_gb)
    requested_drop = read_drop_list(args.drop_list)
    drop_cols = [col for col in requested_drop if col in df.columns and col not in {"case_id", "target", "WEEK_NUM"}]
    df = df.drop(columns=drop_cols)
    df = reduce_mem_usage(df, use_float16=False)
    check_memory("after drop columns", args.max_rss_gb, args.min_available_gb)

    out_train_path = args.output_run_dir / "train_filtered.parquet"
    df.to_parquet(out_train_path, index=False)
    metadata["train_filtered_path"] = str(out_train_path)
    metadata["variant"] = {
        "base_run_dir": str(args.base_run_dir),
        "drop_list": str(args.drop_list),
        "requested_drop_count": len(requested_drop),
        "applied_drop_count": len(drop_cols),
        "applied_drop_columns": drop_cols,
    }
    metadata["selected_polars_columns"] = [col for col in metadata.get("selected_polars_columns", []) if col not in set(drop_cols)]
    (args.output_run_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_run_dir": str(args.output_run_dir), "applied_drop_count": len(drop_cols), "shape": list(df.shape)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
