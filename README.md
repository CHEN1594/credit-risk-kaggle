# Home Credit Credit Risk Model Stability

Minimal v5 project for the Kaggle Home Credit stability competition.

## Layout

```text
src/          reusable feature, preprocessing, metric, and memory code
scripts/      local command-line entry points
submission/   notebook and model artifact to upload/use on Kaggle
references/   known working reference examples
outputs/      local metrics and smoke-test outputs
data/         local Kaggle competition data
```

`src/` is the local source library. The Kaggle notebook is self-contained because `scripts/build_notebook.py` embeds the needed `src/` code into `submission/v5_inference_only.ipynb`.

## Environment

```powershell
conda env create -f environment.yml
conda activate hcrisk
```

## Kaggle Submission

Submit this notebook:

```text
submission/v5_inference_only.ipynb
```

Attach/upload these artifact files as a Kaggle Dataset or Model input:

```text
submission/artifact/model.joblib
submission/artifact/preprocess.json
submission/artifact/feature_columns.json
submission/artifact/selected_polars_columns.json
submission/artifact/v5_manifest.json
```

`preprocess.joblib` is kept locally but the notebook uses `preprocess.json`.

The notebook does not train on Kaggle. It builds hidden-test features, loads the artifact, predicts in parquet batches, and writes `submission.csv`.

## Local Commands

Rebuild the notebook:

```powershell
conda run -n hcrisk python scripts/build_notebook.py
```

Run a local inference smoke test:

```powershell
conda run -n hcrisk python scripts/smoke_inference.py
```

This writes:

```text
outputs/local_smoke_submission.csv
```

Run a rolling-window CV experiment from a filtered feature parquet:

```powershell
conda run -n hcrisk python scripts/cv_features.py --run-dir outputs/experiments/lgbm_v5_medium_ranges_stable_sample300000
```

Exploratory feature analysis scripts live under:

```text
feature_lab/
```

Retrain from local data in two stages:

```powershell
conda run -n hcrisk python scripts/train_features.py --preset medium --features-only --max-rss-gb 30 --min-available-gb 8
conda run -n hcrisk python scripts/train_model.py --run-dir outputs/lgbm_v5_medium_full --n-estimators 1800 --early-stopping-rounds 150 --max-rss-gb 30 --min-available-gb 8
```

The two-stage flow avoids carrying Polars feature-engineering memory into LightGBM training.

## Current Validation

```text
AUC:        0.862669
Gini:       0.725338
Stability:  0.707469
Best iter:  1273
Features:   556
```

Validation uses the last 20 `WEEK_NUM` values from the training data.
