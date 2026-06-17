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

## 时间切分与 case_id 理解

`WEEK_NUM` 可以理解为这一次贷款申请发生在第几周。它不是所有客户共同从 week 0 开始累计的个人生命周期，也不是某个全局统计窗口。

我们做时间验证时，默认假设是：

```text
每个 case_id 的特征 = 这个 case_id 在申请时点可见的信息快照。
```

不是：

```text
所有样本统一使用 week 0 的信息。
```

也不是：

```text
官方在训练集结束后，把每个人截至最后一周的信息统一汇总给了我们。
```

如果 depth=0 是训练集结束后的全局汇总，就会包含未来信息，时间验证会失真，test 也很难自然构造。更合理的理解是：

```text
case_id A：WEEK_NUM = 10  -> 使用第 10 周申请时可见的静态快照。
case_id B：WEEK_NUM = 300 -> 使用第 300 周申请时可见的静态快照。
case_id C：WEEK_NUM = 520 -> 使用第 520 周申请时可见的静态快照。
```

因此，用训练集后 20 个 `WEEK_NUM` 做 holdout，本质是在模拟：

```text
用较早时间的申请训练模型，预测较晚时间的申请。
```

这不会天然造成时间穿越，前提是我们的特征工程只使用该 `case_id` 原始表中已经给出的历史/当前快照信息，不用验证集 target 或未来样本统计去构造训练特征。

`case_id` 也不能理解成永久唯一客户 ID。更准确地说：

```text
一个 case_id = 一次贷款申请 / 一个建模样本。
```

它代表这一次申请唯一，不代表这个真实客户唯一。

所以：

- `base` 里每一行是一笔当前贷款申请。
- 历史表是这个申请相关主体过去或当前可见的金融、征信、人员、税务等记录。
- `target` 是这次申请之后是否违约。
- 官方没有直接给出稳定的 `person_id`，所以不能精确判断两个不同 `case_id` 是否来自同一个真实人。

面试表达：

> 我将 `WEEK_NUM` 理解为申请发生周，将 `depth=0` 理解为申请时点快照，而不是全局结束后的汇总数据；`case_id` 是申请 ID 而不是客户 ID。因此使用后 20 个 `WEEK_NUM` 做时间 holdout，是用历史申请预测未来申请，符合线上 hidden test 的时间外推场景，并不天然引入时间穿越。

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

## 整体建模流程

这个项目的核心流程不是直接把原始数据喂给模型，而是先把多来源、多文件、多 depth 的关系型数据整理成标准机器学习表格。

1. 多表整理与聚合：把来自不同信息源、被拆成多块、且 depth 不同的表，按 `case_id` 进行 join 和聚合，处理成“每个 `case_id` 一行”的单张宽表，方便后续机器学习建模。
2. 特征工程：这一步和多表聚合同时发生。当前 v5 已经做了基础统计聚合，例如 `row_count`、`mean/max/min/std`、`nunique`；时间字段在这里会被转成可聚合的数值特征。后续冲分的重点是继续做更细的业务特征，例如历史申请趋势、逾期模式、征信合同状态、最近行为特征等。
3. 数据预处理：宽表形成之后，再做常规预处理，包括删除 ID/label、类别变量编码、缺失值保留并交给 LightGBM 处理、低信息列过滤、train/test 特征列对齐。
4. 机器学习训练：最后用处理好的宽表训练 LightGBM，并用 `WEEK_NUM` 做时间切分验证；Kaggle 提交时只复用同样的特征工程和预处理逻辑做 inference。

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



## 特征工程的notes

1. 语义出发， 构建有价值的 2. 基础体检 也就是构建前后就需要去看一下这个缺少值啥的， 看看适不适合， 不只是之后要做哦， 构建前也需要哦， 3. 看iv值之类的进行参考。 4.小批量训练看看效果

## 特征工程流程复盘

当前最有效的路线不是盲目堆特征，而是固定验证方式后，按表、按聚合方法、按语义特征组逐步实验。

推荐流程：

```text
粗 baseline
-> 固定 CV
-> 基础清理
-> 按表做实验
-> 按聚合方法做实验
-> 聚合客制化
-> 语义特征构造
-> 按组跑 CV
-> importance / IV / PSI / null importance 辅助筛
-> 再 CV 确认
```

各步骤含义：

1. 粗 baseline：先把 base、depth=0、depth=1、depth=2 的多张表全部压到 `case_id` 级别宽表里，保证信息不漏。缺点是很多聚合很粗，比如日期 sum、类别 mean 这类特征业务意义不强。
2. 固定 CV：固定 fold、模型参数、随机种子、评价指标和训练流程。之后每次只改一个方向，才能判断涨分来自哪里。CV 是本地最终裁判，importance、IV、PSI 都只是辅助。
3. 基础清理：检查全空列、常量列、重复列、几乎全缺失列、聚合后 `case_id` 是否唯一、train/test 类型是否一致。高缺失不一定直接删，因为风控里“缺失本身”可能有信息，可以保留原特征并额外加 missing flag。
4. 按表实验：做表级 ablation，例如去掉某一组 `person`、`bureau_a`、`tax_registry`、`deposit/debitcard` 后看 CV 变化。目的是找重点战场。
5. 按聚合方法实验：看 `sum/max/min/mean/std/first/last/nunique` 哪些有效。不同字段类型不能无脑套同一批聚合。
6. 聚合客制化：按字段含义设计聚合。金额类适合 mean/max/sum/std，日期类更适合 max/min/last 和相对 `date_decision` 的时间差，类别类更适合 nunique/mode/last/count，逾期类更适合 max/count/ratio/recent。
7. 语义特征：主动构造业务上代表风险的变量，例如 DPD 阈值次数、逾期金额、还款比例、收入负债比、拒绝/通过比例、active/closed 合同数量、最近行为等。
8. 按组跑 CV：不要一次性乱加一大坨。按 DPD 组、还款比例组、状态组、最近行为组等分组加入，确认哪一类特征真的有用。
9. 辅助筛选：用 LightGBM importance 看模型是否使用，用 IV/单变量 AUC 看单特征信号，用 PSI 看 train/test 分布稳定性，用 null importance 筛掉“模型喜欢切但不一定有真实信号”的噪声特征。
10. 再 CV 确认：任何删除或新增，最后都要重新跑 CV。辅助指标不能直接决定保留，CV 才能决定。

## 当前项目做到哪里了

| 环节 | 状态 | 说明 |
| --- | --- | --- |
| 粗 baseline | 已完成 | v5 已把主要表压成 `case_id` 宽表，public LB `0.51820`。 |
| 固定 CV | 已完善第一版 | `scripts/cv_features.py` 已改为严格 5-window expanding CV：每折只用验证窗口之前的周 fit 预处理和训练模型，最后输出 mean/min/std/last20 gini。最终提交训练仍用全部训练周。 |
| 基础清理 | 部分完成 | 已做列过滤、类别映射、train/test 对齐、低信息列过滤；重复列、系统 missing flag、高缺失保留策略还没有完全体系化。 |
| 按表实验 | 部分完成 | v8/v9 已围绕 `bureau_a_1`、`person`、`tax`、`deposit/debitcard` 做过方向实验，但还没有完整的逐表 remove/add ablation。 |
| 按聚合方法实验 | 部分完成 | 已发现全表统一聚合会产生大量噪声，并尝试过宽/窄聚合；但还没系统跑过“去掉全部 sum/std/first/last”这类方法级实验。 |
| 聚合客制化 | 已明显推进 | v9 借鉴 example2 的数据处理方式，按表定制聚合，并把日期转成相对 `date_decision` 的时间差；目前 public LB 最好，`0.55357`。 |
| 语义特征 | 部分完成 | 已尝试 A2 DPD 阈值、overdue、active/closed、时间差等特征；部分方向本地有效但没有稳定超过 v9。后续还可继续做还款比例、状态比例、recent 行为等组。 |
| 按组 CV | 部分完成 | 已用小样本和多窗口跑过若干组，但还没有形成每一组特征的稳定实验表。 |
| importance / IV / PSI | 部分完成 | `feature_lab/feature_report.py` 已用于部分 A2 特征体检；但 v9 全量特征还没有系统做 IV/PSI/importance 汇总。 |
| null importance | 未做 | 成本较高，当前优先级低。等特征工程进入精筛阶段再考虑。 |
| 再 CV / LB 确认 | 持续进行 | 当前记录：v5 `0.51820`，v6 `0.54247`，v8 `0.53090`，v9 `0.55357`。 |

## 当前最重要结论

目前正式主线应该以 v9 为 baseline。v9 的价值在于：它不是继续堆更多聚合，而是把聚合方式改得更接近字段语义和 example2 的成熟处理方式，所以 LB 明显超过 v5/v6/v8。

下一步优先级：

1. 以 v9 为基线做表级 ablation，确认哪些表和表组贡献最大。
2. 在 v9 内做聚合方法 ablation，优先删掉业务意义弱、容易过拟合、内存占用高的聚合。
3. 对 v9 全量特征做 importance / IV / PSI 体检，找高收益和高风险特征。
4. 在 v9 上按组加入语义特征，例如 DPD 阈值、还款比例、状态比例、recent 行为，每组单独 CV。
5. 每次只改一个方向，最后用 full holdout 和 Kaggle LB 双重确认。

面试表达可以简化成：

> 我先把 Home Credit 的多源多深度关系表统一压缩成 `case_id` 级宽表，再固定时间切分 CV，逐步做表级和聚合方法 ablation。后续从暴力聚合改成按表、按字段语义定制聚合，例如金额、日期、类别、逾期字段分别采用不同聚合方式，并用 IV、PSI、feature importance 和时间 CV 辅助筛选，最终以 Kaggle LB 做外部验证。
