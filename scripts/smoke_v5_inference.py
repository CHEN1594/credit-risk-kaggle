from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.feature_filter import hash_string_columns, select_polars_columns
from src.features import build_features
from src.memory import check_memory
from src.preprocess import PreprocessState, apply_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test v5 inference artifact locally.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs/lgbm_v5_medium_sample/artifact"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kaggle_v5_smoke"))
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.artifact_dir / "v5_manifest.json").read_text(encoding="utf-8"))
    selected_polars_cols = json.loads(
        (args.artifact_dir / manifest["files"]["selected_polars_columns"]).read_text(encoding="utf-8")
    )
    preprocess_payload = json.loads((args.artifact_dir / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
    state = PreprocessState(
        feature_cols=preprocess_payload["feature_cols"],
        category_maps=preprocess_payload["category_maps"],
        fill_values=preprocess_payload["fill_values"],
        dropped_columns=preprocess_payload["dropped_columns"],
    )
    model = joblib.load(args.artifact_dir / manifest["files"]["model"])
    sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")

    check_memory("before test build", args.max_rss_gb, args.min_available_gb)
    test_pl = build_features(
        args.data_dir,
        "test",
        manifest["preset"],
        cache_dir=args.output_dir / "features",
        use_cache=True,
    )
    test_pl = select_polars_columns(test_pl, [col for col in selected_polars_cols if col != "target"])
    test_pl = hash_string_columns(test_pl, seed=int(manifest.get("string_hash_seed", 42)))
    test_path = args.output_dir / "test_selected.parquet"
    test_pl.write_parquet(test_path)
    del test_pl
    gc.collect()
    check_memory("after write selected test parquet", args.max_rss_gb, args.min_available_gb)

    test_pdf = pd.read_parquet(test_path)
    test_ids = test_pdf["case_id"].to_numpy()
    X_test = apply_preprocessor(test_pdf, state, use_float16=bool(manifest.get("use_float16", False)))
    del test_pdf
    gc.collect()

    matrix_path = args.output_dir / "test_matrix.csv"
    X_test.to_csv(matrix_path, index=False)
    del X_test
    gc.collect()
    check_memory("after write test matrix", args.max_rss_gb, args.min_available_gb)

    pred_chunks = []
    for chunk in pd.read_csv(matrix_path, chunksize=args.chunk_size):
        pred_chunks.append(model.predict_proba(chunk)[:, 1].astype(np.float32))
        del chunk
    predictions = np.concatenate(pred_chunks)

    submission = pd.DataFrame({"case_id": test_ids, "score": predictions})
    submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
    submission["score"] = submission["score"].fillna(float(np.mean(predictions)))
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    print(submission.shape)
    print(submission.head())


if __name__ == "__main__":
    main()
