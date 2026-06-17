from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an inference-only LightGBM submission notebook.")
    parser.add_argument("--output", type=Path, default=Path("submission/v9_inference_only.ipynb"))
    parser.add_argument("--version-label", default="V9")
    parser.add_argument("--working-dir-name", default="kaggle_v9")
    return parser.parse_args()


args = parse_args()
extra_artifact_candidates = ""
if args.version_label.upper() in {"V10", "V11", "V12"}:
    version_lower = args.version_label.lower()
    extra_artifact_candidates = (
        f'        Path("/kaggle/input/home-credit-{version_lower}-artifacts"),\n'
        f'        Path("/kaggle/input/hcrisk-{version_lower}-artifacts"),\n'
        f'        Path("submission/artifact_{version_lower}"),\n'
    )

features_source = embedded_source("src/features.py")
memory_source = embedded_source("src/memory.py")
preprocess_source = embedded_source("src/preprocess.py")
feature_filter_source = embedded_source("src/feature_filter.py")

cells = [
    markdown_cell(
        """
# {version_label} Inference-Only LightGBM Submission

This notebook does not train a model on Kaggle.

Expected inputs:

- Competition data mounted under `/kaggle/input/...home-credit-credit-risk-model-stability`.
- A separate Kaggle Dataset or Model containing the local v9 artifact directory with:
  - `v5_manifest.json`
  - `model.joblib`
  - `preprocess.json`
  - `selected_polars_columns.json`

The notebook builds test features, including `credit_bureau_a_2` two-stage aggregation and table-specific aggregation rules when requested by the manifest, writes the selected matrix to disk, releases memory, then predicts in chunks.
""".replace("{version_label}", args.version_label)
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
import pyarrow.parquet as pq
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
{extra_artifact_candidates}        Path("/kaggle/input/home-credit-v6-artifacts"),
        Path("/kaggle/input/home-credit-v9-artifacts"),
        Path("/kaggle/input/home-credit-v8-artifacts"),
        Path("/kaggle/input/hcrisk-v6-artifacts"),
        Path("/kaggle/input/hcrisk-v9-artifacts"),
        Path("/kaggle/input/hcrisk-v8-artifacts"),
        Path("outputs/experiments_v9_full/artifact_ex2_data"),
        Path("outputs/experiments_v8_full/artifact_custom_bureau_a1_narrow"),
        Path("outputs/experiments_v6_twostage/artifact_a2_twostage"),
        Path("submission/artifact_v9"),
        Path("submission/artifact_v8"),
        Path("submission/artifact_v6"),
        Path("submission/artifact"),
    ]
    for candidate in candidates:
        if (candidate / "v5_manifest.json").exists():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for manifest_path in input_root.rglob("v5_manifest.json"):
            return manifest_path.parent
    for manifest_path in Path("submission").rglob("v5_manifest.json"):
        return manifest_path.parent
    raise FileNotFoundError("Could not find model artifact with v5_manifest.json.")


KAGGLE_ROOT = next((path for path in KAGGLE_ROOT_CANDIDATES if path.exists()), None)
ROOT = KAGGLE_ROOT if KAGGLE_ROOT is not None else (find_kaggle_data_dir() or find_local_data_dir())
ARTIFACT_DIR = find_artifact_dir()
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/{working_dir_name}")
WORKING.mkdir(parents=True, exist_ok=True)

sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
DRY_RUN = sample_submission.shape[0] == 10
MAX_RSS_GB = 30.0
MIN_AVAILABLE_GB = 8.0
BATCH_SIZE = 100_000

print({
    "root": str(ROOT),
    "artifact_dir": str(ARTIFACT_DIR),
    "working": str(WORKING),
    "dry_run": DRY_RUN,
    "sample_submission_rows": int(sample_submission.shape[0]),
})
""".replace("{working_dir_name}", args.working_dir_name).replace("{extra_artifact_candidates}", extra_artifact_candidates)
    ),
    code_cell(memory_source + "\n\nlog_memory('initialized')"),
    code_cell(features_source),
    code_cell(preprocess_source),
    code_cell(feature_filter_source),
    code_cell(
        """
manifest = json.loads((ARTIFACT_DIR / "v5_manifest.json").read_text(encoding="utf-8"))
preprocess_payload = json.loads((ARTIFACT_DIR / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
state = PreprocessState(
    feature_cols=preprocess_payload["feature_cols"],
    category_maps=preprocess_payload["category_maps"],
    fill_values=preprocess_payload["fill_values"],
    dropped_columns=preprocess_payload["dropped_columns"],
    missing_indicator_cols=preprocess_payload.get("missing_indicator_cols", {}),
)
PRESET = manifest["preset"]
FEATURE_SET = manifest.get("feature_set", "none")
USE_FLOAT16 = bool(manifest.get("use_float16", False))
STRING_HASH_SEED = int(manifest.get("string_hash_seed", 42))

print({
    "manifest_version": manifest.get("version"),
    "model_kind": manifest.get("model_kind"),
    "preset": PRESET,
    "feature_set": FEATURE_SET,
    "n_features": len(state.feature_cols),
    "best_iteration": manifest.get("best_iteration"),
})
check_memory("after load metadata", MAX_RSS_GB, MIN_AVAILABLE_GB)
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
    feature_set=FEATURE_SET,
    output_columns=["case_id", *state.feature_cols],
)
print("raw test shape:", test_pl.shape)
check_memory("after test build", MAX_RSS_GB, MIN_AVAILABLE_GB)

test_pl = select_polars_columns(test_pl, ["case_id", *state.feature_cols])
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
files = manifest["files"]
blend = manifest.get("blend", {})
lightgbm_weight = float(blend.get("lightgbm_weight", 1.0))
xgboost_weight = float(blend.get("xgboost_weight", 0.0))
catboost_weight = float(blend.get("catboost_weight", 0.0))

lightgbm_model = joblib.load(ARTIFACT_DIR / files.get("lightgbm_model", files["model"]))
xgboost_model = None
catboost_model = None
if "xgboost_model" in files and (ARTIFACT_DIR / files["xgboost_model"]).exists():
    xgboost_model = joblib.load(ARTIFACT_DIR / files["xgboost_model"])
if "catboost_model" in files and (ARTIFACT_DIR / files["catboost_model"]).exists():
    catboost_model = joblib.load(ARTIFACT_DIR / files["catboost_model"])

weight_total = lightgbm_weight
if xgboost_model is not None:
    weight_total += xgboost_weight
if catboost_model is not None:
    weight_total += catboost_weight
if weight_total <= 0:
    raise ValueError("Invalid blend weights.")

print({
    "lightgbm_weight": lightgbm_weight,
    "xgboost_weight": xgboost_weight if xgboost_model is not None else 0.0,
    "catboost_weight": catboost_weight if catboost_model is not None else 0.0,
    "weight_total": weight_total,
})
check_memory("after load models", MAX_RSS_GB, MIN_AVAILABLE_GB)

pred_chunks = []
id_chunks = []
parquet_file = pq.ParquetFile(test_path)
for idx, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE), start=1):
    check_memory(f"before preprocess batch {idx}", MAX_RSS_GB, MIN_AVAILABLE_GB)
    batch_pdf = batch.to_pandas()
    id_chunks.append(batch_pdf["case_id"].to_numpy())
    X_batch = apply_preprocessor(batch_pdf, state, use_float16=USE_FLOAT16)
    del batch_pdf, batch
    gc.collect()
    check_memory(f"before predict batch {idx}", MAX_RSS_GB, MIN_AVAILABLE_GB)
    pred = lightgbm_weight * lightgbm_model.predict_proba(X_batch)[:, 1]
    if xgboost_model is not None:
        pred = pred + xgboost_weight * xgboost_model.predict_proba(X_batch)[:, 1]
    if catboost_model is not None:
        pred = pred + catboost_weight * catboost_model.predict_proba(X_batch)[:, 1]
    pred = (pred / weight_total).astype(np.float32)
    pred_chunks.append(pred)
    del X_batch, pred
    gc.collect()

predictions = np.concatenate(pred_chunks)
test_ids = np.concatenate(id_chunks)
del pred_chunks, id_chunks, parquet_file, lightgbm_model, xgboost_model, catboost_model
gc.collect()
check_memory("after batch prediction", MAX_RSS_GB, MIN_AVAILABLE_GB)

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

out_path = args.output
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
