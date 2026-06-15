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
preprocess_source = embedded_source("src/preprocess.py")
feature_filter_source = embedded_source("src/feature_filter.py")

cells = [
    markdown_cell(
        """
# V5 Inference-Only LightGBM Submission

This notebook does not train a model on Kaggle.

Expected inputs:

- Competition data mounted under `/kaggle/input/...home-credit-credit-risk-model-stability`.
- A separate Kaggle Dataset containing the local v5 artifact directory with:
  - `v5_manifest.json`
  - `model.joblib`
  - `preprocess.json`
  - `selected_polars_columns.json`

The notebook builds test features, selects only the locally trained feature set, writes the test matrix to disk, releases memory, then predicts in chunks.
"""
    ),
    code_cell(
        """
from __future__ import annotations

import gc
import json
from pathlib import Path
import warnings

import joblib
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
        if (data_dir / "parquet_files" / "test" / "test_base.parquet").exists():
            return data_dir
    return None


def find_artifact_dir() -> Path:
    candidates = [
        Path("/kaggle/input/home-credit-v5-artifacts"),
        Path("/kaggle/input/hcrisk-v5-artifacts"),
        Path("outputs/lgbm_v5_medium_full/artifact"),
        Path("outputs/lgbm_v5_medium_sample/artifact"),
    ]
    for candidate in candidates:
        if (candidate / "v5_manifest.json").exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for manifest_path in input_root.rglob("v5_manifest.json"):
            return manifest_path.parent
    for manifest_path in Path("outputs").rglob("v5_manifest.json"):
        return manifest_path.parent
    raise FileNotFoundError(
        "Could not find v5 model artifact. Upload the local artifact directory as a Kaggle Dataset."
    )


KAGGLE_ROOT = next((path for path in KAGGLE_ROOT_CANDIDATES if path.exists()), None)
ROOT = KAGGLE_ROOT if KAGGLE_ROOT is not None else (find_kaggle_data_dir() or find_local_data_dir())
ARTIFACT_DIR = find_artifact_dir()
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_v5")
WORKING.mkdir(parents=True, exist_ok=True)

sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
DRY_RUN = sample_submission.shape[0] == 10
MAX_RSS_GB = 30.0
MIN_AVAILABLE_GB = 8.0
CHUNK_SIZE = 200_000

print({
    "root": str(ROOT),
    "artifact_dir": str(ARTIFACT_DIR),
    "working": str(WORKING),
    "dry_run": DRY_RUN,
    "sample_submission_rows": int(sample_submission.shape[0]),
})
"""
    ),
    code_cell(memory_source + "\n\nlog_memory('initialized')"),
    code_cell(features_source),
    code_cell(preprocess_source),
    code_cell(feature_filter_source),
    code_cell(
        """
manifest = json.loads((ARTIFACT_DIR / "v5_manifest.json").read_text(encoding="utf-8"))
selected_polars_cols = json.loads((ARTIFACT_DIR / manifest["files"]["selected_polars_columns"]).read_text(encoding="utf-8"))
preprocess_payload = json.loads((ARTIFACT_DIR / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
state = PreprocessState(
    feature_cols=preprocess_payload["feature_cols"],
    category_maps=preprocess_payload["category_maps"],
    fill_values=preprocess_payload["fill_values"],
    dropped_columns=preprocess_payload["dropped_columns"],
)
model = joblib.load(ARTIFACT_DIR / manifest["files"]["model"])
PRESET = manifest["preset"]
USE_FLOAT16 = bool(manifest.get("use_float16", False))
STRING_HASH_SEED = int(manifest.get("string_hash_seed", 42))

print({
    "manifest_version": manifest.get("version"),
    "model_kind": manifest.get("model_kind"),
    "preset": PRESET,
    "n_features": len(state.feature_cols),
    "best_iteration": manifest.get("best_iteration"),
})
check_memory("after load artifact", MAX_RSS_GB, MIN_AVAILABLE_GB)
"""
    ),
    code_cell(
        """
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

test_pl = select_polars_columns(test_pl, [col for col in selected_polars_cols if col != "target"])
test_pl = hash_string_columns(test_pl, seed=STRING_HASH_SEED)
print("selected test shape:", test_pl.shape)
check_memory("after polars select", MAX_RSS_GB, MIN_AVAILABLE_GB)

test_path = WORKING / "test_selected.parquet"
test_pl.write_parquet(test_path)
del test_pl
gc.collect()
check_memory("after write selected test parquet", MAX_RSS_GB, MIN_AVAILABLE_GB)
"""
    ),
    code_cell(
        """
test_pdf = pd.read_parquet(test_path)
test_ids = test_pdf["case_id"].to_numpy()
X_test = apply_preprocessor(test_pdf, state, use_float16=USE_FLOAT16)
del test_pdf
gc.collect()
print("X_test shape:", X_test.shape)
check_memory("after test preprocess", MAX_RSS_GB, MIN_AVAILABLE_GB)

matrix_path = WORKING / "test_matrix.csv"
X_test.to_csv(matrix_path, index=False)
del X_test
gc.collect()
check_memory("after write test matrix and release pandas", MAX_RSS_GB, MIN_AVAILABLE_GB)
"""
    ),
    code_cell(
        """
pred_chunks = []
reader = pd.read_csv(matrix_path, chunksize=CHUNK_SIZE)
for idx, chunk in enumerate(reader, start=1):
    check_memory(f"before predict chunk {idx}", MAX_RSS_GB, MIN_AVAILABLE_GB)
    pred = model.predict_proba(chunk)[:, 1].astype(np.float32)
    pred_chunks.append(pred)
    del chunk, pred
    gc.collect()

predictions = np.concatenate(pred_chunks)
del pred_chunks, reader, model
gc.collect()
check_memory("after chunk prediction", MAX_RSS_GB, MIN_AVAILABLE_GB)

submission = pd.DataFrame({"case_id": test_ids, "score": predictions})
submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
submission["score"] = submission["score"].fillna(float(np.mean(predictions)))
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)

print(submission.shape)
check_memory("submission done", MAX_RSS_GB, MIN_AVAILABLE_GB)
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

out_path = Path("notebooks/v5_inference_only.ipynb")
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
