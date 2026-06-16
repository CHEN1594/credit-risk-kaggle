# Memo

当前保留的是 v5 inference-only 方案。

## 当前提交物

- Notebook：`submission/v5_inference_only.ipynb`
- 模型 artifact：`submission/artifact/`

需要上传或挂载的 artifact 文件：

- `model.joblib`
- `preprocess.json`
- `feature_columns.json`
- `selected_polars_columns.json`
- `v5_manifest.json`

Kaggle notebook 只做特征工程和推理，不在 Kaggle 上训练模型。

## 当前成绩

- Kaggle public score：`0.51820`
- 本地 last-20-week 验证：
  - AUC：`0.862669`
  - Gini：`0.725338`
  - Stability：`0.707469`
  - best_iteration：`1273`
  - 特征数：`556`

本地验证方式：用训练集最后 20 个 `WEEK_NUM` 作为时间验证集，模拟未来时间段泛化。

## 原始数据结构

这个比赛的原始训练数据不是一张表，而是一组多来源、多层级的表。

- 训练 parquet 文件总数：`31` 个。
- 逻辑表组数量：`17` 个表组。
- 表组通过 `case_id` 关联到 `train_base`。
- 最终建模时，我们把它们处理成“每个 `case_id` 一行”的宽表。

表组按 depth 分为：

- `base`：1 个。
- `depth=0`：2 个表组。
- `depth=1`：10 个表组。
- `depth=2`：4 个表组。
- `depth=3`：0 个表组，本比赛没有 depth=3。

大表会被拆成多个 parquet 文件。例如 `static_0` 拆成 2 个文件，`credit_bureau_a_1` 拆成 4 个文件，`credit_bureau_a_2` 拆成 11 个文件。

## Base 表

`train_base` 是样本索引表，定义训练集中有哪些样本。

主要字段：

- `case_id`：样本唯一 ID。
- `date_decision`：贷款决策日期。
- `WEEK_NUM`：周序号，用于时间切分和稳定性评分。
- `MONTH`：月份。
- `target`：训练标签，表示是否违约。

规模：

- `train_base.parquet`：`1,526,659` 行，`5` 列。

## depth=0 表

含义：一个 `case_id` 基本对应一行静态信息。

表组数量：`2` 个。

- `static_0`：内部静态数据，当前申请、客户、贷款相关静态属性。
- `static_cb_0`：外部征信静态数据，来自 credit bureau 的静态属性。

处理方式：

- 直接按 `case_id` left join 到 base。
- 如果某个 `case_id` 没有对应记录，缺失值保留为 `NaN/null`。
- 缺失值主要交给 LightGBM 原生处理。

## depth=1 表

含义：一个 `case_id` 对应多行历史记录，通过 `num_group1` 区分。

表组数量：`10` 个。

- `applprev_1`：历史贷款申请或历史授信申请。
- `other_1`：其他内部历史信息。
- `tax_registry_a_1`：税务登记机构 A 的外部数据。
- `tax_registry_b_1`：税务登记机构 B 的外部数据。
- `tax_registry_c_1`：税务登记机构 C 的外部数据。
- `credit_bureau_a_1`：征信机构 A 的合同/信用记录级数据。
- `credit_bureau_b_1`：征信机构 B 的合同/信用记录级数据。
- `deposit_1`：存款相关历史信息。
- `person_1`：申请人和关联人的人员信息，`num_group1=0` 通常是主申请人。
- `debitcard_1`：借记卡交易相关历史信息。

真实结构示例：`applprev_1`

```text
case_id | num_group1 | annuity_853A | credamount_590A | ...
2       | 0          | 640.2        | 10000.0
2       | 1          | 1682.4       | 16000.0
6       | 0          | 1773.8       | 15980.0
6       | 1          | 4189.6       | 32000.0
6       | 2          | 1110.4       | 17380.0
```

说明：

- `case_id=2` 有 2 条历史申请。
- `case_id=6` 有 3 条历史申请。
- 不能直接 join，否则 base 的一行会被展开成多行。

当前处理方式：

- 对每个 depth=1 表按 `case_id` groupby。
- 数值列和日期列生成 `mean/max/min/std`。
- 类别列生成 `nunique`。
- 每个表保留 `row_count`。
- 聚合后再 left join 回 base。

示例衍生特征：

```text
applprev_1__row_count
applprev_1__credamount_590A__mean
applprev_1__credamount_590A__max
applprev_1__credamount_590A__min
applprev_1__credamount_590A__std
```

## depth=2 表

含义：一个 `case_id` 对应多个 `num_group1`，每个 `num_group1` 下还有多个 `num_group2`。

表组数量：`4` 个。

- `applprev_2`：历史申请下面更细一层的内部信息。
- `person_2`：人员下面更细一层的信息，例如就业记录或关联人明细。
- `credit_bureau_a_2`：征信机构 A 的付款/还款明细级数据。
- `credit_bureau_b_2`：征信机构 B 的付款/还款明细级数据。

真实结构示例：`credit_bureau_a_2`

```text
case_id | num_group1 | num_group2 | pmts_dpd_1073P | pmts_overdue_1140A | pmts_month_158T
388     | 0          | 0          |                |                   | 2.0
388     | 0          | 1          |                |                   | 3.0
388     | 0          | 2          |                |                   | 4.0
388     | 0          | 5          | 0.0            | 0.0               | 7.0
388     | 1          | 0          |                |                   | 2.0
```

说明：

- `num_group1` 可以理解为合同、历史对象或人员对象。
- `num_group2` 是该对象下面的更细明细，例如付款期、就业记录、关联人明细等。

当前处理方式：

- 正式 v5 对已使用的 depth=2 表仍然直接按 `case_id` 聚合。
- 没有做复杂的两级聚合。
- 聚合方法与 depth=1 类似：`row_count`、数值/日期 `mean/max/min/std`、类别 `nunique`。

## 当前 v5 使用的数据

当前正式 v5 使用 `medium` 表集。

已使用：

- base：`train_base`
- depth=0：`static_0`、`static_cb_0`
- depth=1：`person_1`、`applprev_1`、`debitcard_1`、`deposit_1`、`other_1`、`tax_registry_a_1`、`tax_registry_b_1`、`tax_registry_c_1`、`credit_bureau_a_1`、`credit_bureau_b_1`
- depth=2：`person_2`、`applprev_2`、`credit_bureau_b_2`

暂未使用：

- `credit_bureau_a_2`

原因：

- `credit_bureau_a_2` 是最大的 depth=2 表，训练集总行数超过 `1.9` 亿行。
- 它信息量可能很高，但 OOM 风险最大。
- 当前 v5 为了保证 Kaggle 可运行，暂时没有纳入正式提交版。

## feature_definitions.csv

`data/feature_definitions.csv` 是所有原始 predictor 的字段解释表，不只包含 depth=0。

它覆盖：

- depth=0 原始字段。
- depth=1 原始字段。
- depth=2 原始字段。

它解释的是原始字段，例如：

- `credamount_590A`：历史申请的贷款金额或卡额度。
- `pmts_dpd_1073P`：active contract 的 payment 逾期天数。
- `birth_259D`：人员出生日期。

它不直接解释我们衍生后的字段。例如：

```text
applprev_1__credamount_590A__mean
```

这个衍生字段需要理解为：

```text
applprev_1 表中 credamount_590A 字段按 case_id 聚合后的 mean。
```

## 当前特征工程总结

当前正式 v5 的特征工程可以概括为：

- base/depth=0：直接按 `case_id` left join。
- depth=1：按 `case_id` 聚合成统计特征后 join。
- depth=2：已使用的表按 `case_id` 聚合成统计特征后 join。
- 缺失值：保留为 `NaN/null`，主要由 LightGBM 原生处理。
- 类别特征：训练时记录类别映射，推理时复用。
- 内存控制：通过列模板、批量推理、主动内存检查降低 OOM 风险。

可以用于简历表达的版本：

> 处理 Home Credit 风控比赛中 31 个 parquet 文件、17 个多来源表组，覆盖 depth=0/1/2 的静态、历史申请、人员、税务、征信和交易数据；将多层级历史记录按 `case_id` 聚合为稳定宽表特征，并构建本地训练、Kaggle inference-only 的内存安全提交流程。

## 项目结构

- `src/`：可复用源码，被训练脚本、smoke test、notebook builder 共用。
- `scripts/`：本地可执行命令入口。
- `submission/`：最终 notebook 和模型 artifact。
- `references/`：保留的 example 和 example2。
- `feature_lab/`：探索脚本，例如 IV/PSI。
- `outputs/`：本地 metrics、实验结果和 smoke 输出。

Kaggle notebook 不能直接 import 本地 `src/`，所以 `scripts/build_notebook.py` 会把 `src/` 里的必要代码嵌入到 notebook。

## 2026-06-16 特征实验

- 新增多窗口 CV：`scripts/cv_features.py`。
- 新增探索目录：`feature_lab/`。
- 300k 样本、3 个 20-week 窗口结果：
  - baseline `none`：mean stability `0.3037`，min `0.0626`。
  - `ranges_stable`：mean stability `0.3488`，min `0.1434`，当前多窗口最好。
  - `missing`、`counts`、`ratios`、`last`、`ranges_stable+last` 均未通过多窗口筛选。
- full `ranges_stable` 训练：
  - Stability `0.7067`，略低于当前 submission artifact 的 `0.7075`。
  - 因此暂不覆盖 `submission/artifact`。
- 结论：`ranges_stable` 方向值得继续细化，但当前正式提交仍保持原 v5 artifact。
