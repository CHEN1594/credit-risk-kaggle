from __future__ import annotations

import json
from pathlib import Path


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.strip("\n").splitlines(keepends=True)


base_path = Path("notebooks/medium_lgbm_submission.ipynb")
if not base_path.exists():
    raise FileNotFoundError("Run scripts/make_kaggle_notebook.py first.")

nb = json.loads(base_path.read_text(encoding="utf-8"))

set_source(
    nb["cells"][0],
    """
# Medium LightGBM Ensemble Submission v2

Self-contained Kaggle Code Competition notebook for Home Credit - Credit Risk Model Stability.

This version is more memory-conscious than v1: it logs memory, trains seed models sequentially, releases the training matrix before building hidden-test features, and averages predictions across seeds during submission.
""",
)

cell1 = "".join(nb["cells"][1]["source"])
cell1 = cell1.replace(
    'warnings.filterwarnings("ignore")',
    '''warnings.filterwarnings("ignore")


def log_memory(label: str) -> None:
    try:
        import os
        import psutil

        process_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        available_gb = psutil.virtual_memory().available / 1024**3
        print(f"[memory] {label} | rss={process_mb:.1f} MB | available={available_gb:.1f} GB")
    except Exception as exc:
        print(f"[memory] {label} | unavailable: {exc}")


def check_memory(label: str, max_rss_gb: float = 30.0, min_available_gb: float = 8.0) -> None:
    try:
        import os
        import psutil

        process_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        available_gb = psutil.virtual_memory().available / 1024**3
        log_memory(label)
        if process_gb > max_rss_gb:
            raise MemoryError(f"process RSS {process_gb:.1f} GB > {max_rss_gb:.1f} GB")
        if available_gb < min_available_gb:
            raise MemoryError(f"available memory {available_gb:.1f} GB < {min_available_gb:.1f} GB")
    except ImportError:
        log_memory(label)''',
)
cell1 = cell1.replace(
    "SEED = 42\n",
    'SEEDS = [42] if DRY_RUN else [42, 2024]\nMAX_RSS_GB = 30.0\nMIN_AVAILABLE_GB = 8.0\n',
)
cell1 = cell1.replace(
    '"sample_submission_rows": len(sample_submission),\n',
    '"sample_submission_rows": len(sample_submission),\n    "seeds": SEEDS,\n',
)
set_source(nb["cells"][1], cell1)

cell3 = "".join(nb["cells"][3]["source"])
cell3 = cell3.replace(
    "def align_columns(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:\n"
    "    missing = [c for c in reference.columns if c not in frame.columns]",
    "def align_columns(feature_cols: list[str], frame: pd.DataFrame) -> pd.DataFrame:\n"
    "    missing = [c for c in feature_cols if c not in frame.columns]",
)
cell3 = cell3.replace(
    "extra = [c for c in frame.columns if c not in reference.columns]",
    "extra = [c for c in frame.columns if c not in feature_cols]",
)
cell3 = cell3.replace("return frame[reference.columns]", "return frame[feature_cols]")
set_source(nb["cells"][3], cell3)

cell4 = "".join(nb["cells"][4]["source"])
cell4 = cell4.replace(
    'print("train shape:", train_pl.shape)',
    'print("train shape:", train_pl.shape)\ncheck_memory("after build train features", MAX_RSS_GB, MIN_AVAILABLE_GB)',
)
cell4 = cell4.replace(
    "gc.collect()\n\nDROP_COLUMNS",
    'gc.collect()\ncheck_memory("after train to pandas", MAX_RSS_GB, MIN_AVAILABLE_GB)\n\nDROP_COLUMNS',
)
cell4 = cell4.replace(
    'X = train_pdf[feature_cols]',
    'X = train_pdf[feature_cols]\ncheck_memory("after encoding train matrix", MAX_RSS_GB, MIN_AVAILABLE_GB)',
)
set_source(nb["cells"][4], cell4)

set_source(
    nb["cells"][5],
    """
base_params = {
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
    "n_jobs": -1,
    "device_type": "cpu",
    "verbosity": -1,
}

holdout_models = []
valid_pred = np.zeros(len(y_valid), dtype=np.float32)
seed_metrics = []
best_iterations = {}

for seed in SEEDS:
    check_memory(f"before fit holdout seed={seed}", MAX_RSS_GB, MIN_AVAILABLE_GB)
    params = dict(base_params)
    params["random_state"] = seed
    holdout_model = lgb.LGBMClassifier(**params)
    holdout_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
    )

    pred = holdout_model.predict_proba(X_valid)[:, 1].astype(np.float32)
    valid_pred += pred / len(SEEDS)
    best_iteration = int(holdout_model.best_iteration_ or N_ESTIMATORS)
    best_iterations[seed] = best_iteration
    seed_metrics.append(
        {
            "seed": seed,
            "best_iteration": best_iteration,
            "auc": float(roc_auc_score(y_valid, pred)),
            "gini": float(gini_score(y_valid, pred)),
        }
    )
    holdout_models.append(holdout_model)
    check_memory(f"after fit holdout seed={seed}", MAX_RSS_GB, MIN_AVAILABLE_GB)

metrics = {
    "auc": float(roc_auc_score(y_valid, valid_pred)),
    "gini": float(gini_score(y_valid, valid_pred)),
    "valid_start_week": int(valid_start),
    "valid_rows": int(len(y_valid)),
    "train_rows": int(len(y_train)),
    "seeds": SEEDS,
    "best_iterations": best_iterations,
    "seed_metrics": seed_metrics,
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
""",
)

set_source(
    nb["cells"][6],
    """
final_models = []

if DRY_RUN:
    final_models = holdout_models
else:
    del X_train, X_valid, y_train, y_valid, holdout_models
    gc.collect()
    for seed in SEEDS:
        check_memory(f"before fit final seed={seed}", MAX_RSS_GB, MIN_AVAILABLE_GB)
        params = dict(base_params)
        params["random_state"] = seed
        params["n_estimators"] = best_iterations[seed]
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X, y)
        final_models.append(final_model)
        gc.collect()
        check_memory(f"after fit final seed={seed}", MAX_RSS_GB, MIN_AVAILABLE_GB)

del X, y, train_pdf
gc.collect()
check_memory("after releasing train matrix", MAX_RSS_GB, MIN_AVAILABLE_GB)
print("final models ready:", len(final_models))
""",
)

set_source(
    nb["cells"][7],
    """
test_pl = build_features("test", PRESET)
print("test shape:", test_pl.shape)
check_memory("after build test features", MAX_RSS_GB, MIN_AVAILABLE_GB)
test_pdf = to_pandas(test_pl)
del test_pl
gc.collect()

test_ids = test_pdf["case_id"].to_numpy()
test_features = test_pdf.drop(columns=[c for c in DROP_COLUMNS if c in test_pdf.columns])
X_test = align_columns(feature_cols, test_features)
apply_category_maps(X_test, category_maps)
del test_pdf, test_features
gc.collect()
check_memory("after build test matrix", MAX_RSS_GB, MIN_AVAILABLE_GB)

test_pred = np.zeros(len(X_test), dtype=np.float32)
for i, model in enumerate(final_models):
    check_memory(f"predict model {i + 1}/{len(final_models)}", MAX_RSS_GB, MIN_AVAILABLE_GB)
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / len(final_models)

submission = pd.DataFrame({"case_id": test_ids, "score": test_pred})
submission = sample_submission[["case_id"]].merge(submission, on="case_id", how="left")
submission["score"] = submission["score"].fillna(float(np.mean(test_pred)))
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv("submission.csv", index=False)

check_memory("done", MAX_RSS_GB, MIN_AVAILABLE_GB)
print(submission.shape)
submission.head()
""",
)

out_path = Path("notebooks/medium_lgbm_ensemble_v2.ipynb")
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
