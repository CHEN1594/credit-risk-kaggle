# v13 Experiment Notes

记录时间：2026-06-18

## 背景

v11 是当前线上最好版本，public LB 为 `0.56081`。  
v12 是基于 OOF 搜索后调整融合权重的版本，public LB 为 `0.55885`，和 v11 接近但没有更好。

v12 之后继续做了一轮特征工程实验，主要补高手流程里的两块：

1. 表级 ablation / 聚合方法 ablation。
2. 第二轮语义特征构造与筛选。

这轮实验本地 CV 是小幅正向，但提升不大，所以暂时不作为当前主线提交版本，先记录下来。

## 1. 表级 Ablation

基准为 v9 特征体系：

```text
feature_set = a2_twostage + ex2_data
model       = LightGBM
CV          = strict expanding-window CV
```

v9 本地 CV：

```text
mean_gini   0.69355
min_gini    0.66613
last20_gini 0.73947
```

删除各表后的结果：

```text
drop_person_all        mean_gini 0.67855
drop_credit_bureau_a_1 mean_gini 0.68063
drop_credit_bureau_a_2 mean_gini 0.69081
drop_applprev_1        mean_gini 0.69224
drop_tax_all           mean_gini 0.69270
drop_deposit_debitcard mean_gini 0.69361
```

结论：

- `person` 和 `credit_bureau_a_1` 是强表，删除后明显大跌。
- `credit_bureau_a_2` 和 `applprev_1` 有贡献，应继续保留。
- `tax` 偏弱但仍有信息。
- `deposit/debitcard` 对 mean gini 几乎无伤，但 last20 下降，后续可考虑简化，不建议直接删除。

## 2. 聚合方法 Ablation

删除不同聚合方法后的结果：

```text
drop_first_last mean_gini 0.67947
drop_std        mean_gini 0.69261
drop_median     mean_gini 0.69217
drop_count      mean_gini 0.69252
drop_sum_std    mean_gini 0.69235
drop_sum        mean_gini 0.69361
```

结论：

- `first/last` 非常重要，不能全删。
- `std/median/count` 也有贡献，不能粗暴全删。
- `sum` 是最边缘的一类，mean 略高但 last20 略低；后续更合理的做法是按字段类型选择性保留。
- 整体结论是：不能靠“整类删除聚合”大幅提升，还是要做按表、按字段语义的精细化聚合。

## 3. 第二轮语义特征

新增了一组 `semantic_v2` 特征，主要围绕强表和风控语义：

```text
person:
  年龄、收入/贷款/年金/负债比例

credit_bureau_a_1:
  最大 DPD
  最大逾期次数
  最大逾期金额
  逾期金额 / 未偿债务
  债务 / 额度
  信用历史跨度

applprev_1:
  历史最大 DPD
  DPD / 历史申请数
  历史授信 / 当前授信比例

credit_bureau_a_2:
  A2 最大 DPD
  active / closed 最大 DPD 差值
  DPD 阈值比例
  逾期金额 / 还款记录数
```

全量 `semantic_v2` CV：

```text
mean_gini   0.69381
min_gini    0.66763
last20_gini 0.73715
```

解释：

- mean/min 比 v9 略好。
- last20 下降，说明全量语义特征里有噪声项。

随后用 IV / PSI / 单变量 AUC 做筛选，删除低 IV 或高 PSI 的弱项，形成 `semantic_v2_stable_from_full`。

筛选后的 CV：

```text
mean_gini   0.69498
min_gini    0.66860
last20_gini 0.74086
```

相对 v9：

```text
mean_gini   +0.00143
min_gini    +0.00247
last20_gini +0.00140
```

## 4. 当前判断

这轮实验方向是正的，但幅度不大。

`semantic_v2_stable_from_full` 在本地多窗口 CV 上优于 v9，但提升大约只有 `0.001` 到 `0.0025` 量级。考虑 Kaggle public/private 波动，这种提升可能在 LB 上体现，也可能被噪声吃掉。

因此当前暂时不把 v13 作为主线提交版本。

后续如果要回收这部分成果，建议两种方式：

1. 用 `semantic_v2_stable_from_full` 训练单模型，看是否能稳定超过 v10。
2. 把 `semantic_v2_stable_from_full` 接进 v11 的 LGB/XGB/Cat 融合体系，验证融合是否能吃到这部分小幅增益。

## 5. 相关输出位置

```text
表级 ablation:
outputs/experiments_v13_ablation/table_ablation_all_summary.csv

聚合 ablation:
outputs/experiments_v13_ablation/agg_core/
outputs/experiments_v13_ablation/agg_extra/

semantic_v2 full 特征:
outputs/experiments_v13_semantic/lgbm_v5_a2lite_a2_twostage+ex2_data+semantic_v2_full/

semantic_v2 stable 变体:
outputs/experiments_v13_semantic/lgbm_v5_a2lite_a2_twostage+ex2_data+semantic_v2_stable_from_full/

semantic_v2 stable CV:
outputs/experiments_v13_semantic/cv5_semantic_v2_stable_from_full.json

语义特征 IV / PSI 报告:
outputs/experiments_v13_semantic/semantic_v2_feature_report.csv
```
