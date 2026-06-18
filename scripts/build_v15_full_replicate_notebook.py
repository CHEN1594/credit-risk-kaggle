from __future__ import annotations

import argparse
import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def embedded_source(path: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip() != "from __future__ import annotations"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v15 first-place-style ensemble notebook.")
    parser.add_argument("--output", type=Path, default=Path("submission/v15_full_replicate_metric_hack.ipynb"))
    return parser.parse_args()


args = parse_args()
features_source = embedded_source("src/features.py")
memory_source = embedded_source("src/memory.py")
preprocess_source = embedded_source("src/preprocess.py")
metric_hack_source = embedded_source("src/metric_hack.py")

cells = [
    markdown_cell(
        """
# V15 First-Place-Style Ensemble + Metric Hack

This notebook uses two feature branches:

- LGB 644 branch: `artifact_v15/lgb644`
- Cat/DNN 661 branch: `artifact_v15/catdnn661`

Blend:

- LGB: 0.40
- DNN: 0.24
- CatBoost: 0.36

Post-processing:

- divide = 0.5
- reduce = 0.03
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
import pyarrow.parquet as pq
import torch
from torch import nn

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


def find_artifact_root() -> Path:
    candidates = [
        Path("/kaggle/input/home-credit-v15-artifacts"),
        Path("/kaggle/input/hcrisk-v15-artifacts"),
        Path("submission/artifact_v15"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(path for path in input_root.rglob("artifact_v15") if path.is_dir())
    for candidate in candidates:
        if (candidate / "lgb644" / "v5_manifest.json").exists() and (candidate / "catdnn661" / "v5_manifest.json").exists():
            return candidate
    raise FileNotFoundError("Could not find v15 artifact root with lgb644 and catdnn661 subdirectories.")


KAGGLE_ROOT = next((p for p in KAGGLE_ROOT_CANDIDATES if p.exists()), None)
ROOT = KAGGLE_ROOT or find_kaggle_data_dir() or find_local_data_dir()
ARTIFACT_ROOT = find_artifact_root()
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_v15_full_replicate")
WORKING.mkdir(parents=True, exist_ok=True)

sample_submission = pd.read_csv(ROOT / "sample_submission.csv")
print({"data_root": str(ROOT), "artifact_root": str(ARTIFACT_ROOT), "working": str(WORKING), "sample_rows": int(len(sample_submission))})
"""
    ),
    code_cell(memory_source),
    code_cell(features_source),
    code_cell(preprocess_source),
    code_cell(metric_hack_source),
    code_cell(
        """
class TabularDNN(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers = []
        in_features = n_features
        for width in hidden:
            layers.extend([nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)])
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_manifest(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / "v5_manifest.json").read_text(encoding="utf-8"))


def load_preprocess_state(artifact_dir: Path, manifest: dict) -> PreprocessState:
    payload = json.loads((artifact_dir / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
    return PreprocessState(
        feature_cols=payload["feature_cols"],
        category_maps=payload["category_maps"],
        fill_values=payload["fill_values"],
        dropped_columns=payload["dropped_columns"],
        missing_indicator_cols=payload.get("missing_indicator_cols", {}),
    )


def prepare_test_matrix(artifact_dir: Path, branch_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = load_manifest(artifact_dir)
    selected_cols = json.loads((artifact_dir / manifest["files"]["selected_polars_columns"]).read_text(encoding="utf-8"))
    test_pl = build_features(
        ROOT,
        "test",
        manifest["preset"],
        cache_dir=WORKING / f"features_{branch_name}",
        use_cache=False,
        feature_set=manifest.get("feature_set", "none"),
        output_columns=selected_cols,
    )
    print({branch_name + "_features": test_pl.shape})
    test_pdf = test_pl.to_pandas()
    ids = test_pdf[["case_id"]].copy()
    state = load_preprocess_state(artifact_dir, manifest)
    X = apply_preprocessor(test_pdf, state, use_float16=bool(manifest.get("use_float16", False)))
    del test_pl, test_pdf
    gc.collect()
    return ids, X, manifest


def predict_sklearn_chunks(model, X: pd.DataFrame, batch_size: int = 100_000) -> np.ndarray:
    chunks = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        chunks.append(model.predict_proba(batch)[:, 1].astype(np.float32))
        del batch
        gc.collect()
    return np.concatenate(chunks)


def predict_dnn_chunks(model_path: Path, X: pd.DataFrame, batch_size: int = 100_000) -> np.ndarray:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    hidden = tuple(int(x) for x in payload["hidden"])
    model = TabularDNN(int(payload["n_features"]), hidden, float(payload["dropout"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    mean = payload["mean"].astype(np.float32)
    std = payload["std"].astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            arr = X.iloc[start : start + batch_size].to_numpy(dtype=np.float32, copy=True)
            arr = (arr - mean) / std
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            logits = model(torch.from_numpy(arr))
            preds.append(torch.sigmoid(logits).numpy().astype(np.float32))
            del arr, logits
            gc.collect()
    return np.concatenate(preds)
"""
    ),
    code_cell(
        """
LGB_DIR = ARTIFACT_ROOT / "lgb644"
CATDNN_DIR = ARTIFACT_ROOT / "catdnn661"

check_memory("before lgb branch")
lgb_ids, X_lgb, lgb_manifest = prepare_test_matrix(LGB_DIR, "lgb644")
lgb_model = joblib.load(LGB_DIR / lgb_manifest["files"].get("lightgbm_model", lgb_manifest["files"]["model"]))
lgb_pred = predict_sklearn_chunks(lgb_model, X_lgb)
del X_lgb, lgb_model
gc.collect()
check_memory("after lgb branch")

cat_ids, X_cat, cat_manifest = prepare_test_matrix(CATDNN_DIR, "catdnn661")
cat_model = joblib.load(CATDNN_DIR / cat_manifest["files"]["catboost_model"])
cat_pred = predict_sklearn_chunks(cat_model, X_cat)
del cat_model
gc.collect()
check_memory("after cat branch")

dnn_pred = predict_dnn_chunks(CATDNN_DIR / cat_manifest["files"]["dnn_model"], X_cat)
del X_cat
gc.collect()
check_memory("after dnn branch")

pred_df = lgb_ids.copy()
pred_df["lgb"] = lgb_pred
pred_df = pred_df.merge(cat_ids.assign(cat=cat_pred, dnn=dnn_pred), on="case_id", how="left")
pred_df["cat"] = pred_df["cat"].fillna(float(np.nanmean(cat_pred)))
pred_df["dnn"] = pred_df["dnn"].fillna(float(np.nanmean(dnn_pred)))
pred_df["score"] = (0.40 * pred_df["lgb"] + 0.36 * pred_df["cat"] + 0.24 * pred_df["dnn"]).clip(0, 1)

submission = sample_submission[["case_id"]].merge(pred_df[["case_id", "score"]], on="case_id", how="left")
submission["score"] = submission["score"].fillna(float(pred_df["score"].mean()))
submission, mh_summary = apply_metric_hack(submission, ROOT, divide=0.5, reduce=0.03)
print({"metric_hack": mh_summary})
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)
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

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {args.output}")
