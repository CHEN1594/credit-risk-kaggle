from __future__ import annotations

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """
# Medium LightGBM Submission

Self-contained Kaggle Code Competition notebook for Home Credit - Credit Risk Model Stability.

The public notebook environment exposes a tiny 10-row test set. In that mode this notebook runs a fast dry run. During Kaggle submission, the hidden test set is mounted and the notebook runs the full medium preset.
"""
    ),
    code_cell(
        """
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from pandas.api.types import is_float_dtype, is_integer_dtype
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

KAGGLE_ROOT_CANDIDATES = [
    Path("/kaggle/input/home-credit-credit-risk-model-stability"),
    Path("/kaggle/input/competitions/home-credit-credit-risk-model-stability"),
]


def find_local_data_dir() -> Path:
    candidates = [Path("data"), Path.cwd() / "data"]
    candidates.extend(parent / "data" for parent in Path.cwd().resolve().parents)
    for candidate in candidates:
        if (candidate / "sample_submission.csv").exists():
            return candidate
    kaggle_inputs = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        kaggle_inputs = [str(path) for path in input_root.glob("*")]
    raise FileNotFoundError(
        "Could not find the Home Credit competition data. "
        "On Kaggle, add the competition dataset to Notebook Inputs. "
        f"Current /kaggle/input entries: {kaggle_inputs}. "
        "Locally, run the notebook from the project root or keep data/sample_submission.csv available."
    )


def find_kaggle_data_dir() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    candidates = []
    for sample_path in input_root.rglob("sample_submission.csv"):
        data_dir = sample_path.parent
        if (data_dir / "parquet_files" / "train" / "train_base.parquet").exists():
            candidates.append(data_dir)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print("Multiple Kaggle data candidates found:", candidates)
        return candidates[0]
    return None


KAGGLE_ROOT = next((path for path in KAGGLE_ROOT_CANDIDATES if path.exists()), None)
if KAGGLE_ROOT is not None:
    ROOT = KAGGLE_ROOT
else:
    ROOT = find_kaggle_data_dir() or find_local_data_dir()
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/kaggle_notebook")
WORKING.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"
SAMPLE_SUBMISSION = ROOT / "sample_submission.csv"

sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
DRY_RUN = sample_submission.shape[0] == 10

PRESET = "static" if DRY_RUN else "medium"
SAMPLE_ROWS = 50000 if DRY_RUN else 0
N_ESTIMATORS = 120 if DRY_RUN else 2000
EARLY_STOPPING_ROUNDS = 30 if DRY_RUN else 150
VALID_WEEKS = 20
SEED = 42

print({
    "root": str(ROOT),
    "working": str(WORKING),
    "dry_run": DRY_RUN,
    "preset": PRESET,
    "sample_rows": SAMPLE_ROWS,
    "sample_submission_rows": len(sample_submission),
})
"""
    ),
    code_cell(
        """
SPECIAL_COLUMNS = {"case_id", "num_group1", "num_group2"}


@dataclass(frozen=True)
class FeaturePreset:
    depth0_groups: tuple[str, ...]
    aggregate_groups: tuple[str, ...]


PRESETS = {
    "static": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(),
    ),
    "medium": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "person_2",
            "applprev_1",
            "applprev_2",
            "debitcard_1",
            "deposit_1",
            "other_1",
            "tax_registry_a_1",
            "tax_registry_b_1",
            "tax_registry_c_1",
            "credit_bureau_a_1",
            "credit_bureau_b_1",
            "credit_bureau_b_2",
        ),
    ),
}


def files_for_group(split: str, group: str) -> list[Path]:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    paths = sorted(base.glob(f"{split}_{group}*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for split={split!r}, group={group!r}")
    return paths


def scan_group(split: str, group: str) -> pl.LazyFrame:
    frames = [pl.scan_parquet(str(path)) for path in files_for_group(split, group)]
    schemas = [frame.collect_schema() for frame in frames]
    target_schema = {}

    for schema in schemas:
        for col, dtype in schema.items():
            if col not in target_schema or target_schema[col] == pl.Null:
                target_schema[col] = dtype
    for key in ("case_id", "num_group1", "num_group2"):
        if key in target_schema:
            target_schema[key] = pl.Int64
    for col, dtype in list(target_schema.items()):
        if dtype == pl.Null:
            target_schema[col] = pl.Utf8

    normalized = []
    for frame, schema in zip(frames, schemas):
        exprs = []
        for col, target_dtype in target_schema.items():
            if col in schema and schema[col] != target_dtype:
                exprs.append(pl.col(col).cast(target_dtype, strict=False).alias(col))
        normalized.append(frame.with_columns(exprs) if exprs else frame)
    return pl.concat(normalized, how="vertical_relaxed")


def normalize_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    exprs = []
    for name, dtype in schema.items():
        if name.endswith("D") or name == "date_decision":
            if dtype.is_numeric():
                continue
            exprs.append(
                pl.col(name)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, strict=False)
                .cast(pl.Int32)
                .alias(name)
            )
    return lf.with_columns(exprs) if exprs else lf


def load_base(split: str) -> pl.LazyFrame:
    return normalize_dates(scan_group(split, "base"))


def load_depth0(split: str, group: str) -> pl.LazyFrame:
    return normalize_dates(scan_group(split, group)).unique(subset=["case_id"], keep="last")


def aggregate_group(split: str, group: str) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(split, group))
    schema = lf.collect_schema()
    aggs = [pl.len().alias(f"{group}__row_count")]

    for col, dtype in schema.items():
        if col in SPECIAL_COLUMNS:
            continue
        out = f"{group}__{col}"
        if dtype.is_numeric() or dtype == pl.Boolean or col.endswith("D"):
            base = pl.col(col).cast(pl.Float64, strict=False)
            aggs.extend(
                [
                    base.mean().alias(f"{out}__mean"),
                    base.max().alias(f"{out}__max"),
                    base.min().alias(f"{out}__min"),
                    base.std().alias(f"{out}__std"),
                ]
            )
        else:
            aggs.append(pl.col(col).n_unique().alias(f"{out}__nunique"))
    return lf.group_by("case_id").agg(aggs)


def build_features(split: str, preset_name: str) -> pl.DataFrame:
    preset = PRESETS[preset_name]
    lf = load_base(split)
    for group in preset.depth0_groups:
        print(f"join depth0: {split} {group}")
        lf = lf.join(load_depth0(split, group), on="case_id", how="left")
    for group in preset.aggregate_groups:
        print(f"join aggregate: {split} {group}")
        lf = lf.join(aggregate_group(split, group), on="case_id", how="left")
    return lf.collect(engine="streaming")
"""
    ),
    code_cell(
        """
def gini_score(y_true, y_pred) -> float:
    return 2.0 * roc_auc_score(y_true, y_pred) - 1.0


def stability_metric(y_true, y_pred, week_num) -> dict[str, float]:
    frame = pd.DataFrame({"target": y_true, "pred": y_pred, "week": week_num})
    weekly = []
    for week, grp in frame.groupby("week", sort=True):
        if grp["target"].nunique() < 2:
            continue
        weekly.append((float(week), gini_score(grp["target"].to_numpy(), grp["pred"].to_numpy())))

    if len(weekly) < 2:
        return {"stability": np.nan, "mean_gini": np.nan, "slope": np.nan, "residual_std": np.nan}

    weeks = np.array([x[0] for x in weekly], dtype=float)
    ginis = np.array([x[1] for x in weekly], dtype=float)
    slope, intercept = np.polyfit(weeks, ginis, deg=1)
    residuals = ginis - (slope * weeks + intercept)
    return {
        "stability": float(np.mean(ginis) + 88.0 * min(0.0, slope) - 0.5 * np.std(residuals)),
        "mean_gini": float(np.mean(ginis)),
        "slope": float(slope),
        "falling_rate": float(min(0.0, slope)),
        "residual_std": float(np.std(residuals)),
        "n_weeks": float(len(weekly)),
    }


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
    maps = {}
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


def align_columns(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in reference.columns if c not in frame.columns]
    if missing:
        frame = pd.concat([frame, pd.DataFrame(np.nan, index=frame.index, columns=missing)], axis=1)
    extra = [c for c in frame.columns if c not in reference.columns]
    if extra:
        frame = frame.drop(columns=extra)
    return frame[reference.columns]
"""
    ),
    code_cell(
        """
train_pl = build_features("train", PRESET)
if SAMPLE_ROWS:
    train_pl = train_pl.sort("case_id").head(SAMPLE_ROWS)
print("train shape:", train_pl.shape)

train_pdf = to_pandas(train_pl)
del train_pl
gc.collect()

DROP_COLUMNS = {"case_id", "target"}
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
max_week = int(train_pdf["WEEK_NUM"].max())
valid_start = max_week - VALID_WEEKS + 1
valid_mask = train_pdf["WEEK_NUM"] >= valid_start

X_train, X_valid = X.loc[~valid_mask], X.loc[valid_mask]
y_train, y_valid = y.loc[~valid_mask], y.loc[valid_mask]
valid_weeks = weeks[valid_mask]

print("X shape:", X.shape)
print("train rows:", len(y_train), "valid rows:", len(y_valid), "valid_start_week:", valid_start)
"""
    ),
    code_cell(
        """
params = {
    "objective": "binary",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.03,
    "num_leaves": 96,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "device_type": "cpu",
    "verbosity": -1,
}

holdout_model = lgb.LGBMClassifier(**params)
holdout_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)],
)

valid_pred = holdout_model.predict_proba(X_valid)[:, 1]
best_iteration = int(holdout_model.best_iteration_ or N_ESTIMATORS)
metrics = {
    "auc": float(roc_auc_score(y_valid, valid_pred)),
    "gini": float(gini_score(y_valid, valid_pred)),
    "valid_start_week": int(valid_start),
    "valid_rows": int(len(y_valid)),
    "train_rows": int(len(y_train)),
    "best_iteration": best_iteration,
}
metrics.update(stability_metric(y_valid.to_numpy(), valid_pred, valid_weeks))
print(json.dumps(metrics, indent=2))

pd.DataFrame(
    {
        "case_id": train_pdf.loc[valid_mask, "case_id"].to_numpy(),
        "WEEK_NUM": valid_weeks,
        "target": y_valid.to_numpy(),
        "prediction": valid_pred,
    }
).to_csv(WORKING / "valid_predictions.csv", index=False)
(WORKING / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
"""
    ),
    code_cell(
        """
final_params = dict(params)
final_params["n_estimators"] = best_iteration
final_model = lgb.LGBMClassifier(**final_params)

if DRY_RUN:
    # Keep public notebook commits quick. Hidden submissions use the final model trained on all rows.
    final_model = holdout_model
else:
    del X_train, X_valid, y_train, y_valid, holdout_model
    gc.collect()
    final_model.fit(X, y)

print("final model ready")
"""
    ),
    code_cell(
        """
test_pl = build_features("test", PRESET)
print("test shape:", test_pl.shape)
test_pdf = to_pandas(test_pl)
del test_pl
gc.collect()

test_ids = test_pdf["case_id"].to_numpy()
test_features = test_pdf.drop(columns=[c for c in DROP_COLUMNS if c in test_pdf.columns])
X_test = align_columns(X, test_features)
apply_category_maps(X_test, category_maps)

test_pred = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"case_id": test_ids, "score": test_pred})
submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
submission["score"] = submission["score"].fillna(float(np.mean(test_pred)))
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)

print(submission.shape)
submission.head()
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/medium_lgbm_submission.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
