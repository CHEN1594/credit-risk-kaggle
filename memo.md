# Memo

当前保留的是 v5 inference-only 方案。

## 提交物

- Notebook：`submission/v5_inference_only.ipynb`
- 模型 artifact：`submission/artifact/`

需要上传/挂载的 artifact：

- `model.joblib`
- `preprocess.json`
- `feature_columns.json`
- `selected_polars_columns.json`
- `v5_manifest.json`

## 本地验证

- AUC：`0.862669`
- Gini：`0.725338`
- Stability：`0.707469`
- best_iteration：`1273`
- 特征数：`556`

验证方式：训练集最后 20 个 `WEEK_NUM` 做时间切分。

## 结构说明

- `src/`：可复用源码，被训练脚本、smoke test、notebook builder 共用。
- `scripts/`：本地可执行命令入口。
- `submission/`：最终 notebook 和模型 artifact。
- `references/`：保留的 example 和 example2。
- `outputs/`：本地 metrics 和 smoke 输出。

Kaggle notebook 不能直接 import 本地 `src/`，所以 `scripts/build_notebook.py` 会把 `src/` 里的必要代码嵌入到 notebook。

## 2026-06-16 特征实验

- 新增多窗口 CV：`scripts/cv_features.py`。
- 新增探索目录：`feature_lab/`。
- 300k 样本、3 个 20-week 窗口结果：
  - baseline `none`：mean stability `0.3037`，min `0.0626`。
  - `ranges_stable`：mean stability `0.3488`，min `0.1434`，目前多窗口最好。
  - `missing`、`counts`、`ratios`、`last`、`ranges_stable+last` 均未通过多窗口筛选。
- full `ranges_stable` 训练：
  - Stability `0.7067`，略低于当前 submission artifact 的 `0.7075`。
  - 因此暂不覆盖 `submission/artifact`。
- 结论：`ranges_stable` 方向值得后续继续细化，但当前正式提交仍保持原 v5 artifact。
