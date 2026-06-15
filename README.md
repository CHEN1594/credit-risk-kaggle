# Home Credit Credit Risk Model Stability

Local training pipeline for the Kaggle Home Credit stability competition.

## Environment

Create the isolated conda environment:

```powershell
conda env create -f environment.yml
conda activate hcrisk
```

The current environment used by this project is `hcrisk`.

## Data

Expected layout:

```text
data/
  parquet_files/
    train/
    test/
  csv_files/
    train/
    test/
  feature_definitions.csv
  sample_submission.csv
```

The local `test` files are only a small mock test set. Kaggle replaces them with the hidden test set during notebook submission.

## Train

Fast smoke test:

```powershell
conda run -n hcrisk python scripts/train_lgbm.py --preset static --sample-rows 50000 --n-estimators 100
```

Baseline run:

```powershell
conda run -n hcrisk python scripts/train_lgbm.py --preset baseline
```

Baseline smoke run already verified on 50,000 rows:

```text
preset=baseline
auc=0.6834
gini=0.3668
stability=0.3117
validation=last 20 WEEK_NUM values in the 50k-row sample
```

Formal local run before Kaggle submission:

```powershell
conda run -n hcrisk python scripts/train_lgbm.py --preset baseline --n-estimators 2000 --early-stopping-rounds 150
```

GPU LightGBM can be attempted on machines with compatible OpenCL support:

```powershell
conda run -n hcrisk python scripts/train_lgbm.py --preset baseline --device gpu
```

If GPU LightGBM fails on Windows, use CPU first. This competition is usually more constrained by feature construction and validation quality than by a single model's GPU training speed.

Outputs are written to:

```text
outputs/lgbm_<preset>/
  metrics.json
  model.joblib
  valid_predictions.csv
  submission.csv
```

Memory-conscious seed ensemble:

```powershell
conda run -n hcrisk python scripts/train_lgbm_ensemble.py --preset medium --seeds 42,2024 --skip-refit
```

The ensemble script has memory guards:

```text
--max-rss-gb 30
--min-available-gb 8
```

If the Python process exceeds 30 GB RSS or system available memory drops below 8 GB, the script stops itself. On a 40 GB RAM machine, keep these defaults unless the machine is otherwise idle.

## Presets

- `static`: joins only depth=0 static tables. Useful for debugging.
- `baseline`: static tables plus smaller historical tables with generic aggregation.
- `medium`: baseline plus `credit_bureau_a_1`.
- `full`: also includes the very large credit bureau A tables. This is expensive and should be run after the baseline pipeline is validated.

Validation is a time split on the last `--valid-weeks` of `WEEK_NUM`, and reports AUC, gini, and the local stability metric.

## Next Experiments

1. Run the full baseline without `--sample-rows`.
2. Tune `--valid-weeks` around 8, 12, 20, and 30 to compare temporal robustness.
3. Add `--preset full` once the baseline is stable; this includes the very large credit bureau A tables.
4. Add CatBoost/XGBoost scripts and blend validation/test predictions.
5. For Kaggle submission, copy or adapt this pipeline into a notebook that writes `/kaggle/working/submission.csv`.

## Kaggle Notebooks

- `notebooks/medium_lgbm_submission.ipynb`: single-seed medium LightGBM submission.
- `notebooks/medium_lgbm_ensemble_v2.ipynb`: two-seed medium LightGBM ensemble with memory logging and guard checks.
- `notebooks/medium_a2ultra_blend_v3.ipynb`: sequential medium + a2ultra submission blend. This avoids holding both feature matrices at once.
- `notebooks/v4_filtered_medium.ipynb`: filtered single-model medium submission intended to reduce Kaggle OOM risk.

Use v1 first if you need the safest submission. Use v2 after v1 has produced a public score.
