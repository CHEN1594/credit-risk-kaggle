# Plan

## 当前 baseline 状态

当前可提交版本是 `v5_inference_only`。

- 提交方式：Kaggle notebook 只做特征工程和推理，不在 Kaggle 上训练。
- 模型：本地训练的 LightGBM。
- Kaggle public score：`0.51820`。
- 本地验证：按 `WEEK_NUM` 做 last-20-week 时间切分。
- 当前特征表：把多张原始表按 `case_id` 聚合和 join，形成每个 `case_id` 一行的宽表。

当前已经覆盖的数据：

- `base`：已覆盖。
- `depth=0`：全部覆盖。
- `depth=1`：全部覆盖。
- `depth=2`：覆盖了 `applprev_2`、`person_2`、`credit_bureau_b_2`。

当前主要缺口：

- `credit_bureau_a_2` 暂未进入正式 v5。
- 这是最大的 depth=2 表，训练集超过 `1.9` 亿行。
- 它包含征信机构 A 的付款/还款明细，信息价值可能很高，但 OOM 风险最大。

当前 v5 的定位：

- 这是一个能跑通、能提交、不容易 OOM 的 baseline。
- 它完成了基础多表聚合、常规预处理、LightGBM 训练和 inference-only 提交流程。
- 它还不是充分利用全部数据的高分版本。

## v6 目标：补齐 credit_bureau_a_2

v6 的核心目标是把 `credit_bureau_a_2` 安全地纳入特征工程，让所有逻辑表组都被覆盖。

### v6-0：基础 case_id 聚合 OOM 测试

目的：先确认 `credit_bureau_a_2` 在 Kaggle hidden test 上做特征工程时不会 OOM。

做法：

- 新建 feature-only notebook。
- 不训练模型。
- 不加载正式模型。
- 对 `credit_bureau_a_2` 先做最基础的按 `case_id` 聚合。
- 输出 `submission.csv` 时所有分数填 `0.5`。

基础聚合方向：

- 逐文件读取 `credit_bureau_a_2_*`。
- 只选择关键列。
- 按 `case_id` 聚合。
- 生成 `count/max/min/mean/std/range` 等基础特征。
- 每处理完一个文件就释放内存。

成功标准：

- Kaggle notebook 能完整跑完。
- 不发生 OOM。
- 能正常生成 `submission.csv`。

### v6-1：credit_bureau_a_2 两级聚合

目的：真正利用 depth=2 的层级结构，而不是只粗暴按 `case_id` 聚合。

两级结构：

```text
payment-level 明细
  -> case_id + num_group1 合同级聚合
  -> case_id 客户级聚合
```

第一层：按 `case_id + num_group1` 聚合，得到合同级特征。

候选特征：

- 每个合同的付款记录数。
- 每个合同最大逾期天数。
- 每个合同平均逾期天数。
- 每个合同逾期金额总和。
- 每个合同最大逾期金额。
- 每个合同最近付款月份/年份。
- 每个合同是否出现过逾期。

第二层：按 `case_id` 聚合，得到客户级特征。

候选特征：

- 合同数量。
- 有逾期合同数量。
- 合同级最大逾期的 `max/mean/std`。
- 合同级逾期金额总和。
- 平均每个合同付款期数。
- 最大付款期数。

成功标准：

- feature-only notebook 在 Kaggle 上不 OOM。
- 本地能够生成包含 `credit_bureau_a_2` 两级聚合特征的训练宽表。

### v6-2：训练 v6 模型并提交

当 v6-1 的 feature-only 测试确认内存安全后，训练正式 v6 模型。

目标：

- 使用所有逻辑表组。
- 包含 `credit_bureau_a_2` 的两级聚合特征。
- 生成新的 inference-only notebook 和 artifact。
- 提交 Kaggle，获得 v6 public score。

判断标准：

- 如果 v6 public score 明显高于 `0.51820`，说明补齐 `credit_bureau_a_2` 有效。
- 如果提升不明显，再检查是否是聚合方式、特征选择或模型参数问题。

## v7-v9 目标：精细特征工程

在 v6 完成全表覆盖后，后续版本重点做更细的业务特征，而不是只堆通用统计量。

候选方向：

- 时间窗口特征。
- 最近行为特征。
- 趋势特征。
- 逾期模式特征。
- active/closed contract 分开统计。
- 当前申请金额与历史申请金额的比例。
- 历史最大逾期与最近逾期对比。
- 按 `WEEK_NUM` 做稳定性筛选。
- IV / PSI / feature importance 辅助筛选。

原则：

- 每次只引入一组明确的特征方向。
- 先本地时间验证，再决定是否提交。
- 注意内存峰值，不能回到 OOM 状态。

## v10 目标：模型融合

模型融合放在特征工程基本到位之后再做。

候选方向：

- 多 seed LightGBM。
- 不同特征集 LightGBM。
- LightGBM + CatBoost。
- 不同时间窗口验证下的模型融合。
- 概率平均或 rank 平均。

原则：

- 先把单模型特征工程上限做高。
- 再用模型融合补稳定性和泛化。
- 不要过早增加提交复杂度。

## 当前下一步

下一步优先做：

```text
v6-0：credit_bureau_a_2 基础 case_id 聚合 feature-only OOM 测试
```

在这个测试通过之前，不开始正式训练 v6。
