# Memo

## 当前已经完成

- 已建立独立 conda 环境：`hcrisk`。
- 已补充比赛说明到 `instruction.md`。
- 已实现本地训练项目结构：
  - `src/features.py`：多表读取、join、聚合特征。
  - `src/metric.py`：gini 和 stability metric。
  - `scripts/train_lgbm.py`：LightGBM 训练、验证、预测。
  - `scripts/export_importance.py`：导出特征重要性。
- 已生成 Kaggle 可提交 notebook：
  - `notebooks/medium_lgbm_submission.ipynb`
- notebook 已能在本地和 Kaggle notebook 手动运行。
- Kaggle 路径已兼容：
  - `/kaggle/input/home-credit-credit-risk-model-stability`
  - `/kaggle/input/competitions/home-credit-credit-risk-model-stability`

## 不同运行阶段的数据

### 本地运行

- 训练集：完整 `train_*`，有 `target`。
- 测试集：本地 mock test，只有 10 行，没有 `target`。
- 不能得到官方分数，只能用训练集内部切 validation 算本地分数。

### Kaggle notebook 手动 Run

- 训练集：完整 `train_*` 可用。
- 测试集：Kaggle mock test，只有 10 行。
- notebook 会自动进入 `DRY_RUN=True`：
  - 只用前 50,000 行训练数据。
  - 只用 `static` 特征。
  - 用于确认 notebook 能跑通。
- 这个阶段没有官方 public score。

### Kaggle Save Version / Submit

- 训练集：完整 `train_*`。
- 测试集：hidden test，大约是训练集 case 数的 90%。
- notebook 会自动进入正式模式：
  - 使用完整训练集。
  - 使用 `medium` 特征。
  - 生成 `/kaggle/working/submission.csv`。
- 这时才会得到 Kaggle public/private leaderboard 分数。

## 当前分数记录

- Kaggle/notebook dry-run 小版本：
  - 50,000 行训练样本。
  - `static` 特征。
  - Stability：`0.2809`
- 本地完整 medium 验证：
  - 训练行数：`1,314,456`
  - 验证行数：`212,203`
  - 验证方式：最后 20 个 `WEEK_NUM`
  - AUC：`0.8608`
  - Gini：`0.7216`
  - Stability：`0.7017`

## 当前建议

- 先提交 `medium_lgbm_submission.ipynb`，拿一个 Kaggle public score。
- 暂时不要直接跑 `full` preset；之前尝试加入完整 `credit_bureau_a_2` 时内存压力过大。
- 下一步优化应基于 `medium`：
  - 多 seed ensemble。
  - 调整验证窗口。
  - 精选加入 `credit_bureau_a_2` 的少量聚合特征。

## 最新改进记录

- 新增内存监控工具：`src/memory.py`。
- 新增更保守的 ensemble 脚本：`scripts/train_lgbm_ensemble.py`。
- 新增 Kaggle v2 notebook：`notebooks/medium_lgbm_ensemble_v2.ipynb`。
- 默认安全阈值：
  - Python 进程 RSS 不超过 `30GB`。
  - 系统可用内存不低于 `8GB`。
- 当前机器总内存约 `40GB`，不要在本地直接跑 `full` 或无保护的 3 seed 大任务。

## 2026-06-15 后续实验

- 新增 `a2lite` preset：在 `medium` 基础上加入 `credit_bureau_a_2` 精选轻量聚合。
  - 5 万行 smoke test：Stability 约 `0.3930`。
  - 完整训练特征构建后系统可用内存低于安全阈值，已自动停止。
- 新增 `a2core` preset：减少部分弱表，仍加入 `credit_bureau_a_2` 轻量聚合。
  - 5 万行 smoke test：Stability 约 `0.3808`。
  - 完整运行在 pandas 转换后可用内存约 `8.8GB`，过于贴边，已停止。
- 新增 `a2ultra` preset：`credit_bureau_a_1` 和 `credit_bureau_a_2` 都只保留精选聚合字段。
  - 5 万行 smoke test：Stability 约 `0.3957`。
  - 完整单 seed holdout：
    - AUC：`0.8597`
    - Gini：`0.7194`
    - Stability：`0.6999`
    - 内存峰值约 `19.5GB RSS`，安全。
- 验证预测融合：
  - `medium` 与 `a2ultra` 做 50/50 blend。
  - 本地 validation Stability 从 `0.7017` 提升到 `0.7055`。
- 新增提交 notebook：
  - `notebooks/medium_a2ultra_blend_v3.ipynb`
  - 顺序训练 `medium` 和 `a2ultra`，分别预测 hidden test，再 50/50 融合。

## v4 预处理版本

- 针对 v1 Kaggle OOM，新增 v4：
  - `notebooks/v4_filtered_medium.ipynb`
  - `scripts/train_lgbm_v4.py`
  - `src/preprocess.py`
- v4 做了更细的数据预处理：
  - Polars 阶段先删除高缺失列、常量列、高基数类别列。
  - 过滤后先写 parquet，释放 Polars 内存，再读入 pandas，避免大表转换时双份常驻。
  - pandas 阶段继续做常量列过滤、类别编码、类型压缩。
- 本地完整 v4 medium 验证：
  - 缺失率阈值：`max_missing=0.70`
  - 保留特征数：`554`
  - AUC：`0.8613`
  - Gini：`0.7227`
  - Stability：`0.7050`
  - best_iteration：`1211`
- v4 notebook 已通过本地 dry-run。
- 如果 v3 在 Kaggle 仍然 OOM，优先提交 v4。

## Feature-only OOM 诊断

- 新增 notebook：`notebooks/feature_only_oom_diagnostic.ipynb`。
- 目的：只跑 v4 的特征工程，不训练模型。
- 如果特征工程跑完，会输出全 `0.5` 的 `submission.csv`。
- 用法：提交到 Kaggle 后看最后一条 `[memory]` 日志：
  - 如果卡在 `after train build`，说明训练特征构建太大。
  - 如果卡在 `after test build`，说明 hidden test 特征构建太大。
  - 如果 train/test 都能写出 filtered parquet，说明主要 OOM 可能在 pandas 转换或模型训练阶段。

## v5 Inference-only 版本

- v5 改为本地训练模型，Kaggle 只做 test 特征工程和推理。
- 新增：
  - `scripts/train_lgbm_v5_artifact.py`
  - `scripts/train_lgbm_v5_from_filtered.py`
  - `scripts/make_kaggle_notebook_v5.py`
  - `notebooks/v5_inference_only.ipynb`
- 本地训练拆成两个进程：
  - 第一步只构建 filtered parquet，避免 Polars RSS 残留。
  - 第二步新进程读取 parquet 训练模型并导出 artifact。
- 字符串列在 Polars 阶段用固定 seed hash 成 `int32`，避免 pandas object 列爆内存。
- 完整 v5 medium 本地验证：
  - AUC：`0.8627`
  - Gini：`0.7253`
  - Stability：`0.7075`
  - best_iteration：`1273`
  - 保留特征数：`556`
- 正式 artifact 位于：`outputs/lgbm_v5_medium_full/artifact`。
- v5 notebook 已改为更接近 `example2` 的推理方式：
  - Kaggle 端不再整表读 test parquet 到 pandas。
  - 不再写完整 `test_matrix.csv`。
  - 使用最终模型特征列提前裁剪 test。
  - 延迟加载模型，先做特征落盘，再按 parquet batch 预处理和预测。
