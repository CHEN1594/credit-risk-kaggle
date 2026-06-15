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
from pandas.api.types import is_float_dtype, is_integer_dtype
from sklearn.metrics import roc_auc_score

from src.features import PRESETS, build_features
from src.memory import MemoryLimitExceeded, check_memory, log_memory
from src.metric import gini_score, stability_metric


DROP_COLUMNS = {"case_id", "target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-conscious LightGBM seed ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seeds", type=str, default="42,2024")
    parser.add_argument("--skip-refit", action="store_true", help="Use holdout models for prediction.")
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in pdf.columns:
        if pd.api.types.is_bool_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("int8")
        elif is_float_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("float32")
        elif is_integer_dtype(pdf[col]) and col != "case_id":
            pdf[col] = pd.to_numeric(pdf[col], downcast="integer")
    return pdf


def fit_category_maps(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[object]]:
    maps: dict[str, list[object]] = {}
    for col in columns:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            cat = frame[col].astype("category")
            maps[col] = list(cat.cat.categories)
            frame[col] = cat.cat.codes.astype("int32")
    return maps


def apply_category_maps(frame: pd.DataFrame, maps: dict[str, list[object]]) -> None:
    for col, categories in maps.items():
        if col in frame.columns:
            mapper = {value: idx for idx, value in enumerate(categories)}
            frame[col] = frame[col].map(mapper).fillna(-1).astype("int32")
    for col in frame.columns:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].astype("category").cat.codes.astype("int32")


def align_columns(feature_cols: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1)
    extra = [col for col in frame.columns if col not in feature_cols]
    if extra:
        frame = frame.drop(columns=extra)
    return frame[feature_cols]


def params_for(seed: int, args: argparse.Namespace, n_estimators: int | None = None) -> dict:
    return {
        "objective": "binary",
        "n_estimators": int(n_estimators or args.n_estimators),
        "learning_rate": args.learning_rate,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "device_type": args.device,
        "verbosity": -1,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    run_dir = args.output_dir / f"lgbm_{args.preset}_ensemble"
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    def guard(label: str) -> None:
        check_memory(label, max_rss_gb=args.max_rss_gb, min_available_gb=args.min_available_gb)

    guard("start")
    train = build_features(
        args.data_dir,
        "train",
        args.preset,
        cache_dir=args.output_dir / "features",
        use_cache=not args.no_cache,
        sample_rows=args.sample_rows,
    )
    guard("after build train features")

    train_pdf = to_pandas(train)
    del train
    gc.collect()
    guard("after train to pandas")

    max_week = int(train_pdf["WEEK_NUM"].max())
    valid_start = max_week - args.valid_weeks + 1
    valid_mask = train_pdf["WEEK_NUM"] >= valid_start
    y = train_pdf["target"].astype("int8")
    weeks = train_pdf["WEEK_NUM"].to_numpy()
    feature_cols = [col for col in train_pdf.columns if col not in DROP_COLUMNS]
    category_maps = fit_category_maps(train_pdf, feature_cols)

    for col in feature_cols:
        if is_float_dtype(train_pdf[col]):
            train_pdf[col] = train_pdf[col].astype("float32")
        elif is_integer_dtype(train_pdf[col]):
            train_pdf[col] = pd.to_numeric(train_pdf[col], downcast="integer")
    X = train_pdf[feature_cols]
    guard("after encoding")

    X_train, X_valid = X.loc[~valid_mask], X.loc[valid_mask]
    y_train, y_valid = y.loc[~valid_mask], y.loc[valid_mask]
    valid_weeks = weeks[valid_mask]

    seed_metrics = []
    valid_preds = np.zeros(len(y_valid), dtype=np.float32)
    best_iterations: dict[int, int] = {}

    for seed in seeds:
        guard(f"before fit holdout seed={seed}")
        model = lgb.LGBMClassifier(**params_for(seed, args))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(100)],
        )
        pred = model.predict_proba(X_valid)[:, 1].astype(np.float32)
        valid_preds += pred / len(seeds)
        best_iter = int(model.best_iteration_ or args.n_estimators)
        best_iterations[seed] = best_iter
        seed_metrics.append(
            {
                "seed": seed,
                "best_iteration": best_iter,
                "auc": float(roc_auc_score(y_valid, pred)),
                "gini": float(gini_score(y_valid, pred)),
            }
        )
        if args.skip_refit:
            joblib.dump(model, model_dir / f"model_seed{seed}.joblib")
        del model, pred
        gc.collect()
        guard(f"after fit holdout seed={seed}")

    ensemble_metrics = {
        "auc": float(roc_auc_score(y_valid, valid_preds)),
        "gini": float(gini_score(y_valid, valid_preds)),
        "valid_start_week": int(valid_start),
        "valid_rows": int(len(y_valid)),
        "train_rows": int(len(y_train)),
        "seeds": seeds,
        "best_iterations": best_iterations,
        "seed_metrics": seed_metrics,
    }
    ensemble_metrics.update(stability_metric(y_valid.to_numpy(), valid_preds, valid_weeks))
    print(json.dumps(ensemble_metrics, indent=2))

    pd.DataFrame(
        {
            "case_id": train_pdf.loc[valid_mask, "case_id"].to_numpy(),
            "WEEK_NUM": valid_weeks,
            "target": y_valid.to_numpy(),
            "prediction": valid_preds,
        }
    ).to_csv(run_dir / "valid_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(ensemble_metrics, indent=2), encoding="utf-8")

    if not args.skip_refit:
        del X_train, X_valid, y_train, y_valid
        gc.collect()
        for seed in seeds:
            guard(f"before fit final seed={seed}")
            final_model = lgb.LGBMClassifier(**params_for(seed, args, n_estimators=best_iterations[seed]))
            final_model.fit(X, y)
            joblib.dump(final_model, model_dir / f"model_seed{seed}.joblib")
            del final_model
            gc.collect()
            guard(f"after fit final seed={seed}")

    del X, y, train_pdf
    gc.collect()
    guard("after releasing train matrix")

    test = build_features(
        args.data_dir,
        "test",
        args.preset,
        cache_dir=args.output_dir / "features",
        use_cache=not args.no_cache,
    )
    test_pdf = to_pandas(test)
    del test
    gc.collect()
    test_ids = test_pdf["case_id"].to_numpy()
    test_features = test_pdf.drop(columns=[col for col in DROP_COLUMNS if col in test_pdf.columns])
    X_test = align_columns(feature_cols, test_features)
    apply_category_maps(X_test, category_maps)
    guard("after build test matrix")

    test_pred = np.zeros(len(X_test), dtype=np.float32)
    for seed in seeds:
        guard(f"before predict seed={seed}")
        model = joblib.load(model_dir / f"model_seed{seed}.joblib")
        test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / len(seeds)
        del model
        gc.collect()

    pd.DataFrame({"case_id": test_ids, "score": test_pred}).to_csv(run_dir / "submission.csv", index=False)
    joblib.dump({"features": feature_cols, "category_maps": category_maps}, run_dir / "preprocess.joblib")
    guard("done")
    print(f"Saved run outputs to {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
