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
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.memory import MemoryLimitExceeded, check_memory
from src.metric import gini_score, stability_metric
from src.preprocess import apply_preprocessor, fit_preprocessor, summarize_drops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v5 artifact from a prebuilt filtered train parquet.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/lgbm_v5_medium_full"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("submission/artifact"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=1800)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--use-float16", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--memory-check-interval", type=int, default=25)
    return parser.parse_args()


def params(args: argparse.Namespace, n_estimators: int | None = None) -> dict:
    return {
        "objective": "binary",
        "n_estimators": int(n_estimators or args.n_estimators),
        "learning_rate": 0.03,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "device_type": "cpu",
        "verbosity": -1,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def memory_callback(args: argparse.Namespace, label: str):
    interval = max(1, int(args.memory_check_interval))

    def _callback(env) -> None:
        if env.iteration == 0 or (env.iteration + 1) % interval == 0:
            check_memory(f"{label} iteration {env.iteration + 1}", args.max_rss_gb, args.min_available_gb)

    _callback.order = 80
    return _callback


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.run_dir / "feature_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    def guard(label: str) -> None:
        check_memory(label, args.max_rss_gb, args.min_available_gb)

    guard("start model stage")
    train_pdf = pd.read_parquet(train_path)
    guard("after read filtered train pandas")

    max_week = int(train_pdf["WEEK_NUM"].max())
    valid_start = max_week - args.valid_weeks + 1
    valid_mask = train_pdf["WEEK_NUM"] >= valid_start
    case_ids = train_pdf["case_id"].to_numpy()
    y = train_pdf["target"].astype("int8")
    weeks = train_pdf["WEEK_NUM"].to_numpy()

    state = fit_preprocessor(
        train_pdf,
        max_missing=float(metadata["max_missing"]),
        max_cat_unique=int(metadata["max_cat_unique"]),
        use_float16=args.use_float16,
    )
    X = apply_preprocessor(train_pdf, state, use_float16=args.use_float16)
    del train_pdf
    gc.collect()
    print("X shape:", X.shape)
    print("drop summary:", summarize_drops(state))
    guard("after preprocess")

    X_train, X_valid = X.loc[~valid_mask], X.loc[valid_mask]
    y_train, y_valid = y.loc[~valid_mask], y.loc[valid_mask]
    valid_weeks = weeks[valid_mask]

    holdout_model = lgb.LGBMClassifier(**params(args))
    holdout_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(100),
            memory_callback(args, "holdout fit"),
        ],
    )
    pred = holdout_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    best_iteration = int(holdout_model.best_iteration_ or args.n_estimators)
    metrics = {
        "auc": float(roc_auc_score(y_valid, pred)),
        "gini": float(gini_score(y_valid, pred)),
        "valid_start_week": int(valid_start),
        "valid_rows": int(len(y_valid)),
        "train_rows": int(len(y_train)),
        "best_iteration": int(best_iteration),
        "preset": metadata["preset"],
        "sample_rows": int(metadata["sample_rows"]),
        "max_missing": float(metadata["max_missing"]),
        "max_cat_unique": int(metadata["max_cat_unique"]),
        "polars_drop_summary": {key: len(value) for key, value in metadata["polars_drops"].items()},
        "drop_summary": summarize_drops(state),
        "n_features": len(state.feature_cols),
    }
    metrics.update(stability_metric(y_valid.to_numpy(), pred, valid_weeks))
    print(json.dumps(metrics, indent=2))
    write_json(args.metrics_path, metrics)
    pd.DataFrame(
        {
            "case_id": case_ids[valid_mask],
            "WEEK_NUM": valid_weeks,
            "target": y_valid.to_numpy(),
            "prediction": pred,
        }
    ).to_csv(args.run_dir / "valid_predictions.csv", index=False)

    if args.skip_final:
        final_model = holdout_model
        final_kind = "holdout"
    else:
        del X_train, X_valid, y_train, y_valid, pred
        gc.collect()
        guard("before final fit")
        final_model = lgb.LGBMClassifier(**params(args, n_estimators=best_iteration))
        final_model.fit(X, y, callbacks=[memory_callback(args, "final fit")])
        del holdout_model
        final_kind = "refit_all"
        gc.collect()
        guard("after final fit")

    state_payload = {
        "feature_cols": state.feature_cols,
        "category_maps": state.category_maps,
        "fill_values": state.fill_values,
        "dropped_columns": state.dropped_columns,
    }
    joblib.dump(final_model, artifact_dir / "model.joblib", compress=3)
    joblib.dump(state_payload, artifact_dir / "preprocess.joblib", compress=3)
    write_json(artifact_dir / "preprocess.json", state_payload)
    write_json(artifact_dir / "feature_columns.json", state.feature_cols)
    write_json(artifact_dir / "selected_polars_columns.json", metadata["selected_polars_columns"])
    manifest = {
        "version": "v5",
        "model_kind": final_kind,
        "preset": metadata["preset"],
        "max_missing": float(metadata["max_missing"]),
        "max_cat_unique": int(metadata["max_cat_unique"]),
        "use_float16": bool(args.use_float16),
        "string_hash_seed": int(metadata["string_hash_seed"]),
        "best_iteration": int(best_iteration),
        "metrics": metrics,
        "files": {
            "model": "model.joblib",
            "preprocess": "preprocess.json",
            "feature_columns": "feature_columns.json",
            "selected_polars_columns": "selected_polars_columns.json",
        },
    }
    write_json(artifact_dir / "v5_manifest.json", manifest)
    print("artifact_dir:", artifact_dir)


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
