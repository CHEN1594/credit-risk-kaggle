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
import polars as pl
from sklearn.metrics import roc_auc_score

from src.features import PRESETS, build_features
from src.memory import MemoryLimitExceeded, check_memory
from src.metric import gini_score, stability_metric
from src.preprocess import apply_preprocessor, fit_preprocessor, summarize_drops


def filter_polars_columns(
    df: pl.DataFrame,
    *,
    max_missing: float,
    max_cat_unique: int,
    min_unique: int = 2,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    keep = {"case_id", "target", "WEEK_NUM"}
    dropped: dict[str, list[str]] = {"missing": [], "constant": [], "high_cardinality": []}
    selected: list[str] = []
    n_rows = len(df)
    for col, dtype in zip(df.columns, df.dtypes):
        if col in keep:
            selected.append(col)
            continue
        missing = df[col].null_count() / n_rows
        if missing > max_missing:
            dropped["missing"].append(col)
            continue
        nunique = df[col].n_unique()
        if nunique < min_unique:
            dropped["constant"].append(col)
            continue
        if dtype in (pl.String, pl.Categorical) and nunique > max_cat_unique:
            dropped["high_cardinality"].append(col)
            continue
        selected.append(col)
    return df.select(selected), dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 single-model LightGBM with feature filtering.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=1800)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--max-missing", type=float, default=0.92)
    parser.add_argument("--max-cat-unique", type=int, default=200)
    parser.add_argument("--use-float16", action="store_true")
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / f"lgbm_v4_{args.preset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def guard(label: str) -> None:
        check_memory(label, args.max_rss_gb, args.min_available_gb)

    guard("start")
    train = build_features(
        args.data_dir,
        "train",
        args.preset,
        cache_dir=args.output_dir / "features",
        use_cache=True,
        sample_rows=args.sample_rows,
    )
    guard("after build features")
    train, polars_drops = filter_polars_columns(
        train,
        max_missing=args.max_missing,
        max_cat_unique=args.max_cat_unique,
    )
    print("polars_drop_summary:", {key: len(value) for key, value in polars_drops.items()}, "kept", len(train.columns))
    guard("after polars feature filter")

    filtered_path = run_dir / "train_filtered.parquet"
    train.write_parquet(filtered_path)
    del train
    gc.collect()
    guard("after write filtered parquet")

    train_pdf = pd.read_parquet(filtered_path)
    guard("after read filtered pandas")

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
    guard("after preprocess")

    X_train, X_valid = X.loc[~valid_mask], X.loc[valid_mask]
    y_train, y_valid = y.loc[~valid_mask], y.loc[valid_mask]
    valid_weeks = weeks[valid_mask]

    model = lgb.LGBMClassifier(**params(args))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(100)],
    )
    pred = model.predict_proba(X_valid)[:, 1]
    best_iteration = int(model.best_iteration_ or args.n_estimators)
    metrics = {
        "auc": float(roc_auc_score(y_valid, pred)),
        "gini": float(gini_score(y_valid, pred)),
        "valid_start_week": int(valid_start),
        "valid_rows": int(len(y_valid)),
        "train_rows": int(len(y_train)),
        "best_iteration": best_iteration,
        "polars_drop_summary": {key: len(value) for key, value in polars_drops.items()},
        "drop_summary": summarize_drops(state),
        "n_features": len(state.feature_cols),
    }
    metrics.update(stability_metric(y_valid.to_numpy(), pred, valid_weeks))
    print(json.dumps(metrics, indent=2))

    pd.DataFrame(
        {
            "case_id": case_ids[valid_mask],
            "WEEK_NUM": valid_weeks,
            "target": y_valid.to_numpy(),
            "prediction": pred,
        }
    ).to_csv(run_dir / "valid_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "preprocess": state}, run_dir / "model.joblib")


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
