from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.memory import MemoryLimitExceeded, check_memory
from src.metric import gini_score, stability_metric
from src.preprocess import apply_preprocessor, fit_preprocessor, summarize_drops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling time-window CV on a filtered feature parquet.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--gap-weeks", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary",
        "n_estimators": args.n_estimators,
        "learning_rate": 0.035,
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


def fold_windows(max_week: int, valid_weeks: int, folds: int, gap_weeks: int) -> list[tuple[int, int]]:
    windows = []
    end = max_week
    for _ in range(folds):
        valid_end = end
        valid_start = valid_end - valid_weeks + 1
        if valid_start < 0:
            break
        windows.append((valid_start, valid_end))
        end = valid_start - gap_weeks - 1
    return windows


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    output_path = args.output_path or (args.run_dir / "cv_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    check_memory("before read train parquet", args.max_rss_gb, args.min_available_gb)
    train_pdf = pd.read_parquet(train_path)
    check_memory("after read train parquet", args.max_rss_gb, args.min_available_gb)

    y = train_pdf["target"].astype("int8")
    weeks = train_pdf["WEEK_NUM"].to_numpy()
    max_week = int(train_pdf["WEEK_NUM"].max())
    windows = fold_windows(max_week, args.valid_weeks, args.folds, args.gap_weeks)
    if not windows:
        raise ValueError("No CV windows could be built.")

    state = fit_preprocessor(
        train_pdf,
        max_missing=float(metadata["max_missing"]),
        max_cat_unique=int(metadata["max_cat_unique"]),
        use_float16=bool(metadata.get("use_float16", False)),
    )
    X = apply_preprocessor(train_pdf, state, use_float16=bool(metadata.get("use_float16", False)))
    del train_pdf
    gc.collect()
    check_memory("after preprocess", args.max_rss_gb, args.min_available_gb)

    fold_results = []
    for idx, (valid_start, valid_end) in enumerate(windows, start=1):
        valid_mask = (weeks >= valid_start) & (weeks <= valid_end)
        train_mask = weeks < valid_start
        X_train, X_valid = X.loc[train_mask], X.loc[valid_mask]
        y_train, y_valid = y.loc[train_mask], y.loc[valid_mask]
        valid_week_values = weeks[valid_mask]
        model = lgb.LGBMClassifier(**params(args))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(100)],
        )
        pred = model.predict_proba(X_valid)[:, 1].astype(np.float32)
        result = {
            "fold": idx,
            "valid_start_week": int(valid_start),
            "valid_end_week": int(valid_end),
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "auc": float(roc_auc_score(y_valid, pred)),
            "gini": float(gini_score(y_valid, pred)),
        }
        result.update(stability_metric(y_valid.to_numpy(), pred, valid_week_values))
        fold_results.append(result)
        print(json.dumps(result, indent=2))
        del model, X_train, X_valid, y_train, y_valid, pred
        gc.collect()
        check_memory(f"after fold {idx}", args.max_rss_gb, args.min_available_gb)

    summary = {
        "preset": metadata["preset"],
        "feature_set": metadata.get("feature_set", "none"),
        "n_features": len(state.feature_cols),
        "drop_summary": summarize_drops(state),
        "folds": fold_results,
        "mean_auc": float(np.mean([fold["auc"] for fold in fold_results])),
        "mean_gini": float(np.mean([fold["gini"] for fold in fold_results])),
        "mean_stability": float(np.mean([fold["stability"] for fold in fold_results])),
        "min_stability": float(np.min([fold["stability"] for fold in fold_results])),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
