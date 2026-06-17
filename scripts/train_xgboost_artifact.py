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
import pandas as pd
from xgboost import XGBClassifier

from src.memory import MemoryLimitExceeded, check_memory
from src.preprocess import PreprocessState, apply_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a final XGBoost model on an existing filtered feature parquet.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/experiments_v9_full/lgbm_v5_a2lite_a2_twostage+ex2_data_full"))
    parser.add_argument("--base-artifact-dir", type=Path, default=Path("submission/artifact_v9"))
    parser.add_argument("--output-artifact-dir", type=Path, default=Path("submission/artifact_v10"))
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=80.0)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary:logistic",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "auc",
        "verbosity": 1,
    }


def copy_base_artifact(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "model.joblib",
        "preprocess.joblib",
        "preprocess.json",
        "feature_columns.json",
        "selected_polars_columns.json",
        "v5_manifest.json",
    ]:
        shutil.copy2(base_dir / name, out_dir / name)


def main() -> None:
    args = parse_args()
    check_memory("start xgboost train", args.max_rss_gb, args.min_available_gb)
    metadata = json.loads((args.run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    copy_base_artifact(args.base_artifact_dir, args.output_artifact_dir)
    base_manifest = json.loads((args.base_artifact_dir / "v5_manifest.json").read_text(encoding="utf-8"))
    preprocess_payload = json.loads((args.base_artifact_dir / base_manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
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
    X = apply_preprocessor(train_pdf, state, use_float16=bool(base_manifest.get("use_float16", False)))
    del train_pdf
    gc.collect()
    check_memory("after preprocess", args.max_rss_gb, args.min_available_gb)

    model = XGBClassifier(**params(args))
    model.fit(X, y, verbose=100)
    check_memory("after fit xgboost", args.max_rss_gb, args.min_available_gb)
    joblib.dump(model, args.output_artifact_dir / "xgboost_model.joblib", compress=3)

    v10_manifest = dict(base_manifest)
    v10_manifest["version"] = "v10"
    v10_manifest["model_kind"] = "lgbm_xgboost_blend_refit_all"
    v10_manifest["blend"] = {
        "lightgbm_weight": 0.5,
        "xgboost_weight": 0.5,
        "rationale": "XGBoost improved mean/min expanding-window gini while LightGBM kept the best last20 gini.",
    }
    v10_manifest["xgboost"] = {
        "model": "xgboost_model.joblib",
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "cv_reference": "outputs/experiments_v9_full/cv5_xgboost_ex2_data_full.json",
    }
    v10_manifest["files"] = dict(v10_manifest["files"])
    v10_manifest["files"]["lightgbm_model"] = v10_manifest["files"]["model"]
    v10_manifest["files"]["xgboost_model"] = "xgboost_model.joblib"
    (args.output_artifact_dir / "v10_manifest.json").write_text(json.dumps(v10_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"artifact_dir": str(args.output_artifact_dir), "xgboost_model": "xgboost_model.joblib"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
