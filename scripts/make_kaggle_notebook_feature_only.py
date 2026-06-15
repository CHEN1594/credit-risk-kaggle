from __future__ import annotations

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def embedded_source(path: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if line.strip() != "from __future__ import annotations")


features_source = embedded_source("src/features.py")
memory_source = embedded_source("src/memory.py")

cells = [
    markdown_cell(
        """
# Feature Engineering OOM Diagnostic

This notebook intentionally does **not** train a model.

It runs the v4-style feature engineering on train/test, then writes a constant `0.5` submission. The goal is to isolate whether Kaggle OOM happens during feature engineering rather than model training.

If it fails, the last printed `[memory]` line identifies the stage.
"""
    ),
    code_cell(
        """
from __future__ import annotations

import gc
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

KAGGLE_ROOT_CANDIDATES = [
    Path("/kaggle/input/home-credit-credit-risk-model-stability"),
    Path("/kaggle/input/competitions/home-credit-credit-risk-model-stability"),
]


def find_local_data_dir() -> Path:
    candidates = [Path("data"), Path.cwd() / "data"]
    candidates.extend(parent / "data" for parent in Path.cwd().resolve().parents)
    for candidate in candidates:
        if (candidate / "sample_submission.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find competition data.")


def find_kaggle_data_dir() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    for sample_path in input_root.rglob("sample_submission.csv"):
        data_dir = sample_path.parent
        if (data_dir / "parquet_files" / "train" / "train_base.parquet").exists():
            return data_dir
    return None


KAGGLE_ROOT = next((path for path in KAGGLE_ROOT_CANDIDATES if path.exists()), None)
ROOT = KAGGLE_ROOT if KAGGLE_ROOT is not None else (find_kaggle_data_dir() or find_local_data_dir())
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_feature_only")
WORKING.mkdir(parents=True, exist_ok=True)

sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
DRY_RUN = sample_submission.shape[0] == 10
PRESET = "static" if DRY_RUN else "medium"
SAMPLE_ROWS = 50000 if DRY_RUN else 0
MAX_MISSING = 0.70
MAX_CAT_UNIQUE = 200
MAX_RSS_GB = 30.0
MIN_AVAILABLE_GB = 8.0

print({
    "root": str(ROOT),
    "working": str(WORKING),
    "dry_run": DRY_RUN,
    "preset": PRESET,
    "sample_rows": SAMPLE_ROWS,
})
"""
    ),
    code_cell(memory_source + "\n\nlog_memory('initialized')"),
    code_cell(features_source),
    code_cell(
        """
def filter_polars_columns(
    df: pl.DataFrame,
    *,
    max_missing: float,
    max_cat_unique: int,
    min_unique: int = 2,
) -> tuple[pl.DataFrame, list[str], dict[str, list[str]]]:
    keep = {"case_id", "target", "WEEK_NUM"}
    dropped = {"missing": [], "constant": [], "high_cardinality": []}
    selected = []
    n_rows = len(df)
    for col, dtype in zip(df.columns, df.dtypes):
        if col in keep:
            selected.append(col)
            continue
        missing = df[col].null_count() / n_rows
        if missing > max_missing:
            dropped["missing"].append(col)
            continue
        nunique = df[col].n_unique()
        if nunique < min_unique:
            dropped["constant"].append(col)
            continue
        if dtype in (pl.String, pl.Categorical) and nunique > max_cat_unique:
            dropped["high_cardinality"].append(col)
            continue
        selected.append(col)
    return df.select(selected), selected, dropped


def select_polars_columns(df: pl.DataFrame, selected: list[str]) -> pl.DataFrame:
    available = [col for col in selected if col in df.columns]
    return df.select(available)
"""
    ),
    code_cell(
        """
print("=== train feature engineering ===")
check_memory("before train build", MAX_RSS_GB, MIN_AVAILABLE_GB)
train_pl = build_features(
    ROOT,
    "train",
    PRESET,
    cache_dir=WORKING / "features",
    use_cache=True,
    sample_rows=SAMPLE_ROWS,
)
print("raw train shape:", train_pl.shape)
check_memory("after train build", MAX_RSS_GB, MIN_AVAILABLE_GB)

train_pl, selected_cols, drops = filter_polars_columns(
    train_pl,
    max_missing=MAX_MISSING,
    max_cat_unique=MAX_CAT_UNIQUE,
)
print("filtered train shape:", train_pl.shape)
print("drops:", {key: len(value) for key, value in drops.items()})
check_memory("after train filter", MAX_RSS_GB, MIN_AVAILABLE_GB)

train_path = WORKING / "train_feature_only_filtered.parquet"
train_pl.write_parquet(train_path)
print("wrote:", train_path)
del train_pl
gc.collect()
check_memory("after release train features", MAX_RSS_GB, MIN_AVAILABLE_GB)
"""
    ),
    code_cell(
        """
print("=== test feature engineering ===")
check_memory("before test build", MAX_RSS_GB, MIN_AVAILABLE_GB)
test_pl = build_features(
    ROOT,
    "test",
    PRESET,
    cache_dir=WORKING / "features",
    use_cache=True,
)
print("raw test shape:", test_pl.shape)
check_memory("after test build", MAX_RSS_GB, MIN_AVAILABLE_GB)

test_pl = select_polars_columns(test_pl, [col for col in selected_cols if col != "target"])
print("filtered test shape:", test_pl.shape)
check_memory("after test filter", MAX_RSS_GB, MIN_AVAILABLE_GB)

test_path = WORKING / "test_feature_only_filtered.parquet"
test_pl.write_parquet(test_path)
print("wrote:", test_path)
del test_pl
gc.collect()
check_memory("after release test features", MAX_RSS_GB, MIN_AVAILABLE_GB)
"""
    ),
    code_cell(
        """
submission = sample_submission[["case_id"]].copy()
submission["score"] = 0.5
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)
check_memory("submission done", MAX_RSS_GB, MIN_AVAILABLE_GB)
print(submission.shape)
submission.head()
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/feature_only_oom_diagnostic.ipynb")
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
