# 联合零与近零正结果：全部方法分析交接

状态：COMPLETE，主线程重算PASS。600条冻结原值查询（每规模100对）全部通过实际PG COUNT；81项模型评估16200行、24CPU评估4800行、27PG评估5400计划，共26400原始记录。本地结果完整，原正式训练与主表保持不变。

选择只依据真实计数和固定种子，不读取模型误差。正例为1..10行，零例仅改单个原值谓词，列、操作符和谓词数保持配对一致；每条单列条件都有真实匹配，排除了空值域/NULL比较直接导致的零。每规模200条原值语义唯一。候选不足的规模用固定seed新行锚点补齐，来源和拒绝计数保存在兄弟joint_zero_20261111的manifest。

这是定向稀有结果压力分布，三个规模的查询集合不同，不能用该表独立推断数据规模效应。

## 结果

每个指标先按单run计算，再取重复的中位数。模型为3训练seed×3MC seed；匹配采样3个sample seed；PG3次ANALYZE；精确边际与10000行样本各1次。重复类型不同，不当作相同统计重复，也不标95%CI。

|规模|方法|正例q-error p95|零例估计p95行|正例估计恰为0/100|正例高于配对零例|
|---|---|---:|---:|---:|---:|
|1000000|own_made|3.000|0.269|0|100%|
|1000000|own_resmade|3.003|0.212|0|100%|
|1000000|official_resmade|3.024|0.336|0|100%|
|1000000|exact_marginal_independence|7.023|1.117|0|98%|
|1000000|uniform_sample_10000|10.000|0.000|95|5%|
|1000000|matched_own_resmade|8.000|0.000|58|42%|
|1000000|PG-extstats|4.000|1.000|0|54%|
|5000000|own_made|2.911|0.275|0|99%|
|5000000|own_resmade|3.319|0.253|0|99%|
|5000000|official_resmade|2.996|0.234|0|99%|
|5000000|exact_marginal_independence|7.000|0.597|0|98%|
|5000000|uniform_sample_10000|10.000|0.000|99|1%|
|5000000|matched_own_resmade|10.000|0.000|82|18%|
|5000000|PG-extstats|6.000|1.000|0|56%|
|full|own_made|3.203|0.659|0|99%|
|full|own_resmade|3.145|0.695|0|99%|
|full|official_resmade|2.975|0.709|0|99%|
|full|exact_marginal_independence|9.000|1.402|0|96%|
|full|uniform_sample_10000|10.000|0.000|99|1%|
|full|matched_own_resmade|10.000|0.000|92|8%|
|full|PG-extstats|6.000|1.000|0|60%|

## 解释与限制

- 三种模型能在本定向集合上较好区分稀有正例与配对零例，但输出不是精确0。full各模型的严格分离率均为99%，其中自写ResMADE九次运行均99%；不外推为任意SQL的零识别准确率。
- full均匀10000行样本漏掉99/100正例，匹配ResMADE预算样本的三seed中位数为92/100。真正零例无样本命中不能单独证明优秀；只看零误报指标或floor1整体q-error会掩盖这种退化行为。
- full自写ResMADE仍约22/100正例估计低于1行；floor1 q-error把truth=1且estimate<1都记为1，因此应同时报告原始估计、漏估计数和配对排名。没有把“未恰好输出0”解释成估计准确。
- PG Seq Scan估计常下限为1，正/零配对容易同值。full多列统计严格分离60%，half-tie score为80%；两指标区别保留。
- 精确边际的full配对分离率96%但正例p95为9，说明相对排序与绝对估计精度不同，不能只用分离率判断全面胜负。
- 定向稀有查询中，官方ResMADE的full p95约2.975，自写约3.145，MADE约3.203；与原主表排序不同，不据此挑seed或改主默认配置。
- 模型零结果估计仍有少量超过1行；这些是非空值域联合零，能够补充旧stress_v2只覆盖直接空域的限制。不存在充足证据证明联合零普遍优越。
- 模型/采样逻辑存储比较均排除共同词典；official匹配样本复用own_resmade相同预算。PG目录行字节不是物理存储。GPU同步推断、CPU估计和PG规划/往返计时边界不同；未执行SQL，不声称端到端加速。

## 数据与复建

- `analysis/all_methods_by_run.csv`：132方法run；保留训练/MC/样本/ANALYZE身份、正例指标、零分布、pair分离。
- `analysis/all_methods_configuration_ranges.csv`：30个规模/方法组合，逐run指标的median/min–max。
- `analysis/all_methods_paired_estimates.csv`：13200行配对估计，可画同pair正例/零例散点；这些重复不增加300对独立查询数。
- `analysis/metrics_by_run.csv`、`paired_estimates.csv`：模型81run原始重算；`constant_zero_sanity.csv`只作退化基线诊断，不算竞争模型。
- 本根81个run的`per_query.csv`/status/summary/code为模型原始结果；CPU/PG原始结果在兄弟`joint_zero_20261111/`。
- 正式epoch仍为660条；tiny的240条与本轮推断结果分开，不增加正式训练数。

从项目目录执行（均仅分析，不训练）：

```powershell
python Naru_Lab_20261109/scripts/audit_joint_model_eval.py
python Naru_Lab_20261109/scripts/summarize_joint_comparison.py
```

证据：项目evidence/joint_model_audit.json、joint_comparison_audit.json；CPU/PG分支audit.json及truth_counts.json。后续Agent据此作图和report，无需重跑实验。
