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
    lines = [line for line in lines if line.strip() != "from __future__ import annotations"]
    return "\n".join(lines)


features_source = embedded_source("src/features.py")
memory_source = embedded_source("src/memory.py")

cells = [
    markdown_cell(
        """
# Medium + A2Ultra Blend Submission v3

Memory-conscious final submission notebook.

It trains two LightGBM models sequentially:

- `medium`, fixed at 537 trees
- `a2ultra`, fixed at 868 trees

Predictions are blended 50/50. Local validation blend improved stability from about `0.7017` to `0.7055`.
"""
    ),
    code_cell(
        """
from __future__ import annotations

import gc
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from pandas.api.types import is_float_dtype, is_integer_dtype

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
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_blend_v3")
WORKING.mkdir(parents=True, exist_ok=True)

sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
DRY_RUN = sample_submission.shape[0] == 10
SAMPLE_ROWS = 50000 if DRY_RUN else 0

MAX_RSS_GB = 30.0
MIN_AVAILABLE_GB = 8.0

print({"root": str(ROOT), "working": str(WORKING), "dry_run": DRY_RUN, "sample_rows": SAMPLE_ROWS})
"""
    ),
    code_cell(memory_source + "\n\nlog_memory('initialized')"),
    code_cell(features_source),
    code_cell(
        """
DROP_COLUMNS = {"case_id", "target"}


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in pdf.columns:
        if pd.api.types.is_bool_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("int8")
        elif is_float_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("float32")
        elif is_integer_dtype(pdf[col]) and col != "case_id":
            pdf[col] = pd.to_numeric(pdf[col], downcast="integer")
    return pdf


def fit_category_maps(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[object]]:
    maps = {}
    for col in columns:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            cat = frame[col].astype("category")
            maps[col] = list(cat.cat.categories)
            frame[col] = cat.cat.codes.astype("int32")
    return maps


def apply_category_maps(frame: pd.DataFrame, maps: dict[str, list[object]]) -> None:
    for col, categories in maps.items():
        if col in frame.columns:
            mapper = {value: idx for idx, value in enumerate(categories)}
            frame[col] = frame[col].map(mapper).fillna(-1).astype("int32")
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].astype("category").cat.codes.astype("int32")


def align_columns(feature_cols: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1)
    extra = [col for col in frame.columns if col not in feature_cols]
    if extra:
        frame = frame.drop(columns=extra)
    return frame[feature_cols]


def lgbm_params(n_estimators: int, seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.03,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "device_type": "cpu",
        "verbosity": -1,
    }
"""
    ),
    code_cell(
        """
def train_predict_stage(preset: str, n_estimators: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    print(f"=== stage {preset} n_estimators={n_estimators} ===")
    check_memory(f"{preset}: start", MAX_RSS_GB, MIN_AVAILABLE_GB)

    train_pl = build_features(
        ROOT,
        "train",
        preset,
        cache_dir=WORKING / "features",
        use_cache=True,
        sample_rows=SAMPLE_ROWS,
    )
    print(f"{preset}: train shape", train_pl.shape)
    check_memory(f"{preset}: after build train features", MAX_RSS_GB, MIN_AVAILABLE_GB)

    train_pdf = to_pandas(train_pl)
    del train_pl
    gc.collect()
    check_memory(f"{preset}: after train to pandas", MAX_RSS_GB, MIN_AVAILABLE_GB)

    y = train_pdf["target"].astype("int8")
    feature_cols = [col for col in train_pdf.columns if col not in DROP_COLUMNS]
    category_maps = fit_category_maps(train_pdf, feature_cols)
    for col in feature_cols:
        if is_float_dtype(train_pdf[col]):
            train_pdf[col] = train_pdf[col].astype("float32")
        elif is_integer_dtype(train_pdf[col]):
            train_pdf[col] = pd.to_numeric(train_pdf[col], downcast="integer")
    X = train_pdf[feature_cols]
    check_memory(f"{preset}: before fit", MAX_RSS_GB, MIN_AVAILABLE_GB)

    model = lgb.LGBMClassifier(**lgbm_params(n_estimators=n_estimators, seed=seed))
    model.fit(X, y)

    del X, y, train_pdf
    gc.collect()
    check_memory(f"{preset}: after release train", MAX_RSS_GB, MIN_AVAILABLE_GB)

    test_pl = build_features(
        ROOT,
        "test",
        preset,
        cache_dir=WORKING / "features",
        use_cache=True,
    )
    print(f"{preset}: test shape", test_pl.shape)
    test_pdf = to_pandas(test_pl)
    del test_pl
    gc.collect()

    test_ids = test_pdf["case_id"].to_numpy()
    test_features = test_pdf.drop(columns=[col for col in DROP_COLUMNS if col in test_pdf.columns])
    X_test = align_columns(feature_cols, test_features)
    apply_category_maps(X_test, category_maps)
    del test_pdf, test_features
    gc.collect()
    check_memory(f"{preset}: before predict", MAX_RSS_GB, MIN_AVAILABLE_GB)

    pred = model.predict_proba(X_test)[:, 1].astype(np.float32)
    del model, X_test
    gc.collect()
    check_memory(f"{preset}: done", MAX_RSS_GB, MIN_AVAILABLE_GB)
    return test_ids, pred
"""
    ),
    code_cell(
        """
if DRY_RUN:
    stages = [("static", 71, 42, 1.0)]
else:
    stages = [
        ("medium", 537, 42, 0.5),
        ("a2ultra", 868, 42, 0.5),
    ]

final_ids = None
final_pred = None
for preset, n_estimators, seed, weight in stages:
    ids, pred = train_predict_stage(preset, n_estimators, seed)
    if final_ids is None:
        final_ids = ids
        final_pred = np.zeros(len(pred), dtype=np.float32)
    final_pred += weight * pred

submission = pd.DataFrame({"case_id": final_ids, "score": final_pred})
submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
submission["score"] = submission["score"].fillna(float(np.mean(final_pred)))
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

out_path = Path("notebooks/medium_a2ultra_blend_v3.ipynb")
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
