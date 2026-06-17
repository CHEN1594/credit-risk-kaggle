from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.feature_filter import hash_string_columns, select_polars_columns
from src.features import build_features
from src.memory import check_memory
from src.preprocess import PreprocessState, apply_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test v5 inference artifact locally.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("submission/artifact"))
    parser.add_argument("--output-path", type=Path, default=Path("outputs/local_smoke_submission.csv"))
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_path.parent / "_smoke_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.artifact_dir / "v5_manifest.json").read_text(encoding="utf-8"))
    preprocess_payload = json.loads((args.artifact_dir / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
    state = PreprocessState(
        feature_cols=preprocess_payload["feature_cols"],
        category_maps=preprocess_payload["category_maps"],
        fill_values=preprocess_payload["fill_values"],
        dropped_columns=preprocess_payload["dropped_columns"],
        missing_indicator_cols=preprocess_payload.get("missing_indicator_cols", {}),
    )
    sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")

    check_memory("before test build", args.max_rss_gb, args.min_available_gb)
    test_pl = build_features(
        args.data_dir,
        "test",
        manifest["preset"],
        cache_dir=temp_dir / "features",
        use_cache=False,
        feature_set=manifest.get("feature_set", "none"),
        output_columns=["case_id", *state.feature_cols],
    )
    test_pl = select_polars_columns(test_pl, ["case_id", *state.feature_cols])
    test_pl = hash_string_columns(test_pl, seed=int(manifest.get("string_hash_seed", 42)))
    test_path = temp_dir / "test_selected.parquet"
    test_pl.write_parquet(test_path)
    del test_pl
    gc.collect()
    check_memory("after write selected test parquet", args.max_rss_gb, args.min_available_gb)

    files = manifest["files"]
    blend = manifest.get("blend", {})
    lightgbm_weight = float(blend.get("lightgbm_weight", 1.0))
    xgboost_weight = float(blend.get("xgboost_weight", 0.0))
    catboost_weight = float(blend.get("catboost_weight", 0.0))
    lightgbm_model = joblib.load(args.artifact_dir / files.get("lightgbm_model", files["model"]))
    xgboost_model = None
    catboost_model = None
    if "xgboost_model" in files and (args.artifact_dir / files["xgboost_model"]).exists():
        xgboost_model = joblib.load(args.artifact_dir / files["xgboost_model"])
    if "catboost_model" in files and (args.artifact_dir / files["catboost_model"]).exists():
        catboost_model = joblib.load(args.artifact_dir / files["catboost_model"])
    weight_total = lightgbm_weight
    if xgboost_model is not None:
        weight_total += xgboost_weight
    if catboost_model is not None:
        weight_total += catboost_weight
    pred_chunks = []
    id_chunks = []
    parquet_file = pq.ParquetFile(test_path)
    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        batch_pdf = batch.to_pandas()
        id_chunks.append(batch_pdf["case_id"].to_numpy())
        X_batch = apply_preprocessor(batch_pdf, state, use_float16=bool(manifest.get("use_float16", False)))
        pred = lightgbm_weight * lightgbm_model.predict_proba(X_batch)[:, 1]
        if xgboost_model is not None:
            pred = pred + xgboost_weight * xgboost_model.predict_proba(X_batch)[:, 1]
        if catboost_model is not None:
            pred = pred + catboost_weight * catboost_model.predict_proba(X_batch)[:, 1]
        pred_chunks.append((pred / weight_total).astype(np.float32))
        del batch, batch_pdf, X_batch
    predictions = np.concatenate(pred_chunks)
    test_ids = np.concatenate(id_chunks)

    submission = pd.DataFrame({"case_id": test_ids, "score": predictions})
    submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
    submission["score"] = submission["score"].fillna(float(np.mean(predictions)))
    submission.to_csv(args.output_path, index=False)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(submission.shape)
    print(submission.head())


if __name__ == "__main__":
    main()
