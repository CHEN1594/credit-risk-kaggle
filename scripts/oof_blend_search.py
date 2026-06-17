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
from src.preprocess import apply_preprocessor, fit_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-model OOF predictions and search blend weights.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oof_blend"))
    parser.add_argument("--valid-weeks", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--max-cat-weight", type=float, default=0.3)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fold_windows(max_week: int, valid_weeks: int, folds: int) -> list[tuple[int, int]]:
    windows = []
    end = max_week
    for _ in range(folds):
        valid_end = end
        valid_start = valid_end - valid_weeks + 1
        if valid_start < 0:
            break
        windows.append((valid_start, valid_end))
        end = valid_start - 1
    return list(reversed(windows))


def read_fold_data(train_path: Path, valid_start: int, valid_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_slice = pd.read_parquet(train_path, filters=[("WEEK_NUM", "<", valid_start)])
    valid_slice = pd.read_parquet(
        train_path,
        filters=[
            ("WEEK_NUM", ">=", valid_start),
            ("WEEK_NUM", "<=", valid_end),
        ],
    )
    return train_slice, valid_slice


def lgb_params(args: argparse.Namespace) -> dict:
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


def xgb_params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary:logistic",
        "n_estimators": args.n_estimators,
        "learning_rate": 0.035,
        "max_depth": 6,
        "min_child_weight": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "auc",
        "early_stopping_rounds": args.early_stopping_rounds,
        "verbosity": 1,
    }


def cat_params(args: argparse.Namespace) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": args.n_estimators,
        "learning_rate": 0.035,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_seed": args.seed,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
        "od_type": "Iter",
        "od_wait": args.early_stopping_rounds,
    }


def fit_predict_all(args: argparse.Namespace, X_train, y_train, X_valid, y_valid) -> dict[str, np.ndarray]:
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    preds: dict[str, np.ndarray] = {}

    lgb_model = lgb.LGBMClassifier(**lgb_params(args))
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(100)],
    )
    preds["lgb"] = lgb_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    del lgb_model
    gc.collect()

    xgb_model = XGBClassifier(**xgb_params(args))
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    preds["xgb"] = xgb_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    del xgb_model
    gc.collect()

    cat_model = CatBoostClassifier(**cat_params(args))
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    preds["cat"] = cat_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    del cat_model
    gc.collect()
    return preds


def score_frame(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(frame["target"], pred))
    out = {
        "auc": auc,
        "gini": float(2.0 * auc - 1.0),
    }
    out.update(stability_metric(frame["target"].to_numpy(), pred, frame["WEEK_NUM"].to_numpy()))
    return out


def search_weights(args: argparse.Namespace, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    weights = np.round(np.arange(0.0, 1.0 + 1e-9, args.weight_step), 10)
    for lgb_w in weights:
        for xgb_w in weights:
            cat_w = 1.0 - lgb_w - xgb_w
            if cat_w < -1e-9 or cat_w > args.max_cat_weight + 1e-9:
                continue
            cat_w = round(float(cat_w), 10)
            pred = lgb_w * oof["lgb_pred"].to_numpy() + xgb_w * oof["xgb_pred"].to_numpy() + cat_w * oof["cat_pred"].to_numpy()
            metrics = score_frame(oof, pred)
            rows.append(
                {
                    "lgb_weight": float(lgb_w),
                    "xgb_weight": float(xgb_w),
                    "cat_weight": float(cat_w),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["gini", "stability"], ascending=[False, False])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    week_series = pd.read_parquet(train_path, columns=["WEEK_NUM"])["WEEK_NUM"]
    windows = fold_windows(int(week_series.max()), args.valid_weeks, args.folds)
    del week_series
    gc.collect()

    fold_frames = []
    for fold_idx, (valid_start, valid_end) in enumerate(windows, start=1):
        check_memory(f"before fold {fold_idx} read", args.max_rss_gb, args.min_available_gb)
        train_slice, valid_slice = read_fold_data(train_path, valid_start, valid_end)
        y_train = train_slice["target"].astype("int8")
        y_valid = valid_slice["target"].astype("int8")
        state = fit_preprocessor(
            train_slice,
            max_missing=float(metadata["max_missing"]),
            max_cat_unique=int(metadata["max_cat_unique"]),
            missing_indicator_min_rate=metadata.get("missing_indicator_min_rate"),
            use_float16=bool(metadata.get("use_float16", False)),
        )
        X_train = apply_preprocessor(train_slice, state, use_float16=bool(metadata.get("use_float16", False)))
        X_valid = apply_preprocessor(valid_slice, state, use_float16=bool(metadata.get("use_float16", False)))
        ids = valid_slice["case_id"].to_numpy()
        weeks = valid_slice["WEEK_NUM"].to_numpy()
        del train_slice
        gc.collect()
        check_memory(f"after fold {fold_idx} preprocess", args.max_rss_gb, args.min_available_gb)

        preds = fit_predict_all(args, X_train, y_train, X_valid, y_valid)
        fold_frame = pd.DataFrame(
            {
                "fold": fold_idx,
                "case_id": ids,
                "WEEK_NUM": weeks,
                "target": y_valid.to_numpy(),
                "lgb_pred": preds["lgb"],
                "xgb_pred": preds["xgb"],
                "cat_pred": preds["cat"],
            }
        )
        fold_frames.append(fold_frame)
        print(json.dumps({"fold": fold_idx, "rows": len(fold_frame), "valid_start": valid_start, "valid_end": valid_end}, indent=2))
        del X_train, X_valid, y_train, y_valid, valid_slice, preds, fold_frame
        gc.collect()
        check_memory(f"after fold {fold_idx}", args.max_rss_gb, args.min_available_gb)

    oof = pd.concat(fold_frames, ignore_index=True)
    oof_path = args.output_dir / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    model_rows = []
    for name in ["lgb", "xgb", "cat"]:
        metrics = score_frame(oof, oof[f"{name}_pred"].to_numpy())
        model_rows.append({"model": name, **metrics})
    model_scores = pd.DataFrame(model_rows).sort_values("gini", ascending=False)
    model_scores.to_csv(args.output_dir / "single_model_oof_scores.csv", index=False)

    weights = search_weights(args, oof)
    weights.to_csv(args.output_dir / "blend_weight_search.csv", index=False)
    summary = {
        "oof_path": str(oof_path),
        "rows": int(len(oof)),
        "folds": len(windows),
        "single_model_scores": model_rows,
        "best_weights": weights.head(20).to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
