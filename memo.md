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

