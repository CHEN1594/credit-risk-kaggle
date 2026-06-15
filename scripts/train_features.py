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

from src.feature_filter import filter_polars_columns, hash_string_columns
from src.features import PRESETS, build_features, derived_output_columns
from src.memory import MemoryLimitExceeded, check_memory
from src.metric import gini_score, stability_metric
from src.preprocess import apply_preprocessor, fit_preprocessor, summarize_drops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v5 local model artifacts for Kaggle inference-only submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--feature-set", default="none")
    parser.add_argument("--template-columns", type=Path, default=None)
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=1800)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--max-cat-unique", type=int, default=200)
    parser.add_argument("--use-float16", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final", action="store_true", help="Export the holdout model instead of refitting on all rows.")
    parser.add_argument("--memory-check-interval", type=int, default=25)
    parser.add_argument("--features-only", action="store_true", help="Only build the filtered train parquet and metadata.")
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
    suffix = f"sample{args.sample_rows}" if args.sample_rows else "full"
    run_dir = args.output_dir / f"lgbm_v5_{args.preset}_{args.feature_set}_{suffix}"
    artifact_dir = run_dir / "artifact"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def guard(label: str) -> None:
        check_memory(label, args.max_rss_gb, args.min_available_gb)

    output_columns = None
    if args.template_columns is not None:
        output_columns = json.loads(args.template_columns.read_text(encoding="utf-8"))
        output_columns = list(dict.fromkeys(output_columns + derived_output_columns(args.feature_set)))

    guard("start")
    train_pl = build_features(
        args.data_dir,
        "train",
        args.preset,
        cache_dir=args.output_dir / "features",
        use_cache=True,
        sample_rows=args.sample_rows,
        feature_set=args.feature_set,
        output_columns=output_columns,
    )
    print("raw train shape:", train_pl.shape)
    guard("after build train features")

    train_pl, selected_polars_cols, polars_drops = filter_polars_columns(
        train_pl,
        max_missing=args.max_missing,
        max_cat_unique=args.max_cat_unique,
    )
    print("filtered train shape:", train_pl.shape)
    print("polars_drop_summary:", {key: len(value) for key, value in polars_drops.items()})
    guard("after polars filter")

    train_pl = hash_string_columns(train_pl, seed=args.seed)
    guard("after polars string hash")

    train_path = run_dir / "train_filtered.parquet"
    train_pl.write_parquet(train_path)
    feature_metadata = {
        "version": "v5",
        "preset": args.preset,
        "feature_set": args.feature_set,
        "sample_rows": int(args.sample_rows),
        "max_missing": float(args.max_missing),
        "max_cat_unique": int(args.max_cat_unique),
        "use_float16": bool(args.use_float16),
        "string_hash_seed": int(args.seed),
        "selected_polars_columns": selected_polars_cols,
        "template_columns": output_columns,
        "polars_drops": polars_drops,
        "train_filtered_path": str(train_path),
    }
    write_json(run_dir / "feature_metadata.json", feature_metadata)
    del train_pl
    gc.collect()
    guard("after write filtered train parquet")

    if args.features_only:
        print("features_only_done:", train_path)
        return

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
        max_missing=args.max_missing,
        max_cat_unique=args.max_cat_unique,
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
        "preset": args.preset,
        "sample_rows": int(args.sample_rows),
        "max_missing": float(args.max_missing),
        "max_cat_unique": int(args.max_cat_unique),
        "polars_drop_summary": {key: len(value) for key, value in polars_drops.items()},
        "drop_summary": summarize_drops(state),
        "n_features": len(state.feature_cols),
    }
    metrics.update(stability_metric(y_valid.to_numpy(), pred, valid_weeks))
    print(json.dumps(metrics, indent=2))
    write_json(run_dir / "metrics.json", metrics)

    pd.DataFrame(
        {
            "case_id": case_ids[valid_mask],
            "WEEK_NUM": valid_weeks,
            "target": y_valid.to_numpy(),
            "prediction": pred,
        }
    ).to_csv(run_dir / "valid_predictions.csv", index=False)

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
    write_json(artifact_dir / "selected_polars_columns.json", selected_polars_cols)
    manifest = {
        "version": "v5",
        "model_kind": final_kind,
        "preset": args.preset,
        "feature_set": args.feature_set,
        "max_missing": float(args.max_missing),
        "max_cat_unique": int(args.max_cat_unique),
        "use_float16": bool(args.use_float16),
        "string_hash_seed": int(args.seed),
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
