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
import pandas as pd
from catboost import CatBoostClassifier

from src.memory import MemoryLimitExceeded, check_memory
from src.preprocess import PreprocessState, apply_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a final CatBoost model on an existing filtered feature parquet.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/experiments_v9_full/lgbm_v5_a2lite_a2_twostage+ex2_data_full"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("submission/artifact_v11"))
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_memory("start catboost train", args.max_rss_gb, args.min_available_gb)
    metadata = json.loads((args.run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    manifest_path = args.artifact_dir / "v5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocess_payload = json.loads((args.artifact_dir / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
    state = PreprocessState(
        feature_cols=preprocess_payload["feature_cols"],
        category_maps=preprocess_payload["category_maps"],
        fill_values=preprocess_payload["fill_values"],
        dropped_columns=preprocess_payload["dropped_columns"],
        missing_indicator_cols=preprocess_payload.get("missing_indicator_cols", {}),
    )

    train_pdf = pd.read_parquet(train_path)
    check_memory("after read train parquet", args.max_rss_gb, args.min_available_gb)
    y = train_pdf["target"].astype("int8")
    X = apply_preprocessor(train_pdf, state, use_float16=bool(manifest.get("use_float16", False)))
    del train_pdf
    gc.collect()
    check_memory("after preprocess", args.max_rss_gb, args.min_available_gb)

    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=3.0,
        random_seed=args.seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(X, y)
    check_memory("after fit catboost", args.max_rss_gb, args.min_available_gb)
    joblib.dump(model, args.artifact_dir / "catboost_model.joblib", compress=3)

    manifest["version"] = "v11"
    manifest["model_kind"] = "lgbm_xgboost_catboost_blend_refit_all"
    manifest["blend"] = {
        "lightgbm_weight": 0.45,
        "xgboost_weight": 0.45,
        "catboost_weight": 0.10,
        "rationale": "XGBoost improved mean/min expanding-window gini; LightGBM kept best last20; CatBoost adds diversity with higher stability but lower gini.",
    }
    manifest["files"] = dict(manifest["files"])
    manifest["files"]["lightgbm_model"] = manifest["files"]["model"]
    manifest["files"]["xgboost_model"] = "xgboost_model.joblib"
    manifest["files"]["catboost_model"] = "catboost_model.joblib"
    manifest["catboost"] = {
        "model": "catboost_model.joblib",
        "iterations": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "cv_reference": "outputs/experiments_v9_full/cv5_catboost_ex2_data_full.json",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.artifact_dir / "v11_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"artifact_dir": str(args.artifact_dir), "catboost_model": "catboost_model.joblib"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
