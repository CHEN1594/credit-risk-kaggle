from __future__ import annotations

import argparse
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
from src.metric import gini_score, stability_metric


DROP_COLUMNS = {"case_id", "target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM baseline for Home Credit stability.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="baseline")
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--sample-rows", type=int, default=0, help="Use first N train rows for smoke tests.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    for col in pdf.columns:
        if pd.api.types.is_bool_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("int8")
        elif is_float_dtype(pdf[col]):
            pdf[col] = pdf[col].astype("float32")
        elif is_integer_dtype(pdf[col]) and col not in {"case_id"}:
            pdf[col] = pd.to_numeric(pdf[col], downcast="integer")
    return pdf


def align_test_columns(train_x: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in train_x.columns if c not in test_x.columns]
    if missing:
        test_x = pd.concat(
            [test_x, pd.DataFrame(np.nan, index=test_x.index, columns=missing)],
            axis=1,
        )
    extra = [c for c in test_x.columns if c not in train_x.columns]
    if extra:
        test_x = test_x.drop(columns=extra)
    return test_x[train_x.columns]


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "features"

    train = build_features(
        args.data_dir,
        "train",
        args.preset,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
        sample_rows=args.sample_rows,
    )

    train_pdf = to_pandas(train)
    max_week = int(train_pdf["WEEK_NUM"].max())
    valid_start = max_week - args.valid_weeks + 1
    valid_mask = train_pdf["WEEK_NUM"] >= valid_start

    y = train_pdf["target"].astype(int)
    weeks = train_pdf["WEEK_NUM"].to_numpy()
    feature_cols = [c for c in train_pdf.columns if c not in DROP_COLUMNS]
    category_maps = fit_category_maps(train_pdf, feature_cols)
    for col in feature_cols:
        if is_float_dtype(train_pdf[col]):
            train_pdf[col] = train_pdf[col].astype("float32")
        elif is_integer_dtype(train_pdf[col]):
            train_pdf[col] = pd.to_numeric(train_pdf[col], downcast="integer")
    X = train_pdf[feature_cols]

    X_train, X_valid = X.loc[~valid_mask], X.loc[valid_mask]
    y_train, y_valid = y.loc[~valid_mask], y.loc[valid_mask]
    valid_weeks = weeks[valid_mask]

    params = {
        "objective": "binary",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
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
        "device_type": args.device,
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(50)],
    )

    valid_pred = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_valid, valid_pred)),
        "gini": float(gini_score(y_valid, valid_pred)),
        "valid_start_week": int(valid_start),
        "valid_rows": int(len(y_valid)),
        "train_rows": int(len(y_train)),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
    }
    metrics.update(stability_metric(y_valid.to_numpy(), valid_pred, valid_weeks))

    run_dir = args.output_dir / f"lgbm_{args.preset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": feature_cols, "category_maps": category_maps}, run_dir / "model.joblib")
    pd.DataFrame(
        {
            "case_id": train_pdf.loc[valid_mask, "case_id"].to_numpy(),
            "WEEK_NUM": valid_weeks,
            "target": y_valid.to_numpy(),
            "prediction": valid_pred,
        }
    ).to_csv(run_dir / "valid_predictions.csv", index=False)

    test = build_features(
        args.data_dir,
        "test",
        args.preset,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
    )
    test_pdf = to_pandas(test)
    test_ids = test_pdf["case_id"].to_numpy()
    X_test = align_test_columns(X_train, test_pdf.drop(columns=[c for c in DROP_COLUMNS if c in test_pdf.columns]))
    apply_category_maps(X_test, category_maps)
    test_pred = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"case_id": test_ids, "score": test_pred}).to_csv(run_dir / "submission.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved run outputs to {run_dir}")


if __name__ == "__main__":
    main()
