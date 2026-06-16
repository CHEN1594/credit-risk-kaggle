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
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """
# V6-0 `credit_bureau_a_2` Feature-Only OOM Test

This notebook intentionally does not train or load a model.

It only tests whether the largest depth=2 table, `credit_bureau_a_2`, can be aggregated on Kaggle hidden test without OOM. The final submission uses a constant `0.5` score.
"""
    ),
    code_cell(
        """
from __future__ import annotations

import gc
import os
import sys
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
        if (data_dir / "parquet_files" / "test" / "test_base.parquet").exists():
            return data_dir
    return None


KAGGLE_ROOT = next((path for path in KAGGLE_ROOT_CANDIDATES if path.exists()), None)
ROOT = KAGGLE_ROOT if KAGGLE_ROOT is not None else (find_kaggle_data_dir() or find_local_data_dir())
TEST_DIR = ROOT / "parquet_files" / "test"
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/v6_a2_feature_only")
WORKING.mkdir(parents=True, exist_ok=True)

MAX_RSS_GB = 30.0
MIN_AVAILABLE_GB = 8.0
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

print({
    "root": str(ROOT),
    "test_dir": str(TEST_DIR),
    "working": str(WORKING),
})
"""
    ),
    code_cell(
        """
class MemoryLimitExceeded(RuntimeError):
    pass


def _process_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        pass

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / 1024 if sys.platform.startswith("linux") else rss / 1024**2
        except Exception:
            return None
    return None


def _available_memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def log_memory(label: str) -> None:
    rss = _process_memory_mb()
    available = _available_memory_gb()
    parts = [f"[memory] {label}"]
    if rss is not None:
        parts.append(f"rss={rss:.1f} MB")
    if available is not None:
        parts.append(f"available={available:.1f} GB")
    print(" | ".join(parts))


def check_memory(label: str) -> None:
    rss_mb = _process_memory_mb()
    available_gb = _available_memory_gb()
    log_memory(label)

    if rss_mb is not None and rss_mb / 1024 > MAX_RSS_GB:
        raise MemoryLimitExceeded(
            f"Memory guard stopped at {label}: RSS {rss_mb / 1024:.1f} GB > {MAX_RSS_GB:.1f} GB"
        )
    if available_gb is not None and available_gb < MIN_AVAILABLE_GB:
        raise MemoryLimitExceeded(
            f"Memory guard stopped at {label}: available {available_gb:.1f} GB < {MIN_AVAILABLE_GB:.1f} GB"
        )


check_memory("initialized")
"""
    ),
    code_cell(
        """
def normalize_date_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    exprs = []
    for name, dtype in schema.items():
        if name.endswith("D") or name == "date_decision":
            if dtype.is_numeric():
                continue
            exprs.append(
                pl.col(name)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, strict=False)
                .cast(pl.Int32)
                .alias(name)
            )
    return lf.with_columns(exprs) if exprs else lf


def aggregate_a2_file(path: Path, case_ids: pl.DataFrame, out_path: Path) -> Path:
    print(f"[a2] start {path.name}")
    check_memory(f"before {path.name}")

    scan = pl.scan_parquet(str(path))
    schema = scan.collect_schema()
    available = [col for col in A2_SELECTED_COLUMNS if col in schema]
    if not available:
        raise ValueError(f"No selected A2 columns found in {path}")

    lf = scan.select(["case_id", *available])
    lf = lf.with_columns(pl.col("case_id").cast(pl.Int64))
    lf = lf.join(case_ids.lazy(), on="case_id", how="semi")

    aggs = [pl.len().alias("credit_bureau_a_2__row_count")]
    for col in available:
        base = pl.col(col).cast(pl.Float64, strict=False)
        prefix = f"credit_bureau_a_2__{col}"
        aggs.extend(
            [
                base.sum().alias(f"{prefix}__sum"),
                base.count().alias(f"{prefix}__count"),
                base.max().alias(f"{prefix}__max"),
                base.min().alias(f"{prefix}__min"),
            ]
        )

    partial = lf.group_by("case_id").agg(aggs).collect(engine="streaming")
    partial.write_parquet(out_path)
    print(f"[a2] wrote {out_path.name}: {partial.shape}")

    del partial, lf, scan
    gc.collect()
    check_memory(f"after {path.name}")
    return out_path


def combine_a2_partials(partial_paths: list[Path]) -> pl.DataFrame:
    if not partial_paths:
        raise FileNotFoundError("No A2 partial parquet files were generated.")

    partial_lf = pl.scan_parquet([str(path) for path in partial_paths])
    schema = partial_lf.collect_schema()

    final_aggs = [
        pl.col("credit_bureau_a_2__row_count").sum().alias("credit_bureau_a_2__row_count")
    ]
    for col in A2_SELECTED_COLUMNS:
        prefix = f"credit_bureau_a_2__{col}"
        sum_col = f"{prefix}__sum"
        count_col = f"{prefix}__count"
        max_col = f"{prefix}__max"
        min_col = f"{prefix}__min"
        if sum_col not in schema:
            continue
        total_count = pl.col(count_col).sum()
        final_aggs.extend(
            [
                (pl.col(sum_col).sum() / total_count).alias(f"{prefix}__mean"),
                pl.col(max_col).max().alias(f"{prefix}__max"),
                pl.col(min_col).min().alias(f"{prefix}__min"),
            ]
        )

    result = partial_lf.group_by("case_id").agg(final_aggs).collect(engine="streaming")
    gc.collect()
    check_memory("after combine A2 partials")
    return result
"""
    ),
    code_cell(
        """
sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
base = normalize_date_columns(pl.scan_parquet(str(TEST_DIR / "test_base.parquet"))).collect(engine="streaming")
case_ids = base.select("case_id").with_columns(pl.col("case_id").cast(pl.Int64))
print({
    "sample_submission_shape": sample_submission.shape,
    "test_base_shape": base.shape,
})
check_memory("after load base")

a2_paths = sorted(TEST_DIR.glob("test_credit_bureau_a_2_*.parquet"))
print({"a2_file_count": len(a2_paths), "a2_files": [path.name for path in a2_paths]})
if not a2_paths:
    raise FileNotFoundError("No test_credit_bureau_a_2_*.parquet files found.")

partial_dir = WORKING / "a2_partials"
partial_dir.mkdir(parents=True, exist_ok=True)
partial_paths = []
for idx, path in enumerate(a2_paths):
    out_path = partial_dir / f"a2_partial_{idx:02d}.parquet"
    partial_paths.append(aggregate_a2_file(path, case_ids, out_path))

a2_features = combine_a2_partials(partial_paths)
print({"a2_features_shape": a2_features.shape})

features = base.join(a2_features, on="case_id", how="left")
print({"final_feature_shape": features.shape})
check_memory("after final feature join")

feature_path = WORKING / "v6_a2_caseid_features.parquet"
features.write_parquet(feature_path)
print({"feature_path": str(feature_path)})

del a2_features, features, base, case_ids
gc.collect()
check_memory("after write final features")
"""
    ),
    code_cell(
        """
submission = sample_submission[["case_id"]].copy()
submission["score"] = 0.5
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)

print({"submission_shape": submission.shape})
check_memory("submission done")
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

out_path = Path("submission/v6_a2_caseid_feature_only.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
