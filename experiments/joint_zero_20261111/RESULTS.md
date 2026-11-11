# 联合零结果与近零正结果：CPU / PostgreSQL结果

状态：COMPLETE，600条冻结查询、24项CPU对照、27项PG统计重建评估已完成。本地独立审计PASS：10200条CSV、5400份PG计划、600次PG实际COUNT及33列直方图对齐。GPU模型评估在独立 `../joint_zero_model_eval_20261111/`，本文件不冒充其结果。

## 查询合同

每规模100对：正例truth1..10；零例只改同一列/操作符的一个原值，谓词数不变。两边全部单列条件都在真实表上有匹配，allowed_codes非空，且每规模200种原值语义唯一。固定seed2026111101起，选择不读模型误差。1M正例100个来自main；5M为80 main+20新锚点；full为41 main+59新锚点。新锚点按固定随机顺序删条件、保留truth窗口和至少5列。600查询全部由实际PG COUNT独立验证。

这是针对稀有结果的配对压力分布，非无偏SQL随机样本，三个规模也不是同一查询集合。不能用本组独立推断规模效应。

## 汇总

每项指标先在单次运行计算，再取重复的中位数；范围见metrics_configuration_ranges.csv。PG重复是ANALYZE，匹配采样重复是sample seed，不能当训练seed或置信区间。配对严格分离率指positive_estimate > zero_estimate，另报ties和half-tie score。

|规模|方法|正例q-error p95|零例估计p95行|正例估计0条/100|配对严格分离率|
|---|---|---:|---:|---:|---:|
|1000000|exact_marginal_independence|7.023|1.117|0|98%|
|1000000|uniform_sample_10000|10.000|0.000|95|5%|
|1000000|matched_own_made|8.537|0.000|65|35%|
|1000000|matched_own_resmade|8.000|0.000|58|42%|
|1000000|PG-default|8.000|3.200|0|33%|
|1000000|PG-target1000|7.287|1.000|0|46%|
|1000000|PG-extstats|4.000|1.000|0|54%|
|5000000|exact_marginal_independence|7.000|0.597|0|98%|
|5000000|uniform_sample_10000|10.000|0.000|99|1%|
|5000000|matched_own_made|10.000|0.000|87|13%|
|5000000|matched_own_resmade|10.000|0.000|82|18%|
|5000000|PG-default|8.000|4.050|0|29%|
|5000000|PG-target1000|7.000|1.000|0|41%|
|5000000|PG-extstats|6.000|1.000|0|56%|
|full|exact_marginal_independence|9.000|1.402|0|96%|
|full|uniform_sample_10000|10.000|0.000|99|1%|
|full|matched_own_made|10.000|0.000|98|2%|
|full|matched_own_resmade|10.000|0.000|92|8%|
|full|PG-default|9.000|4.000|0|41%|
|full|PG-target1000|9.000|1.000|0|52%|
|full|PG-extstats|6.000|1.000|0|60%|

## 解释与边界

- 均匀样本对真正零结果必然无命中，但同时漏掉大量1..10行正例：full匹配ResMADE预算采样的正例零估计中位数92/100，10000行样本为99/100。因此不能只看零例误报率称其最好，必须同时看正例漏估和pair分离。
- PG普通Seq Scan估计通常下限为1，导致正/零配对大量ties。full extstats正例p95为6，严格分离率60%；这比default改善，但不是准确识别全部零。
- 精确边际在本定向样本中配对排序较高（full96%），同时正例p95为9，说明相对排序与绝对基数误差不同；不能用单一分离率判全面最优。
- 样本budget为FP32可训练参数字节对int32行字节，共同词典两侧排除。own/official ResMADE预算相同，对应同样本，不新增独立重复。PG保存的是catalog逻辑行字节，不是物理synopsis分配。
- 计时：CPU只估计，预建mask/hist/sample排除；PG分别服务器planning/client roundtrip，不执行SQL，不能与GPU直接计算SQL加速比。

## 复现与证据

- `fixed_config.json`、`workloads/{scale}/joint_zero_nearzero.jsonl`及manifest：冻结负载与生成策略。
- `pg/{scale}/truth_counts.json`、`marginal_histogram_check.json`：全部600次实际COUNT、33列PG原值直方图。
- `pg/{scale}/{tier}/repeat*/`：5400完整JSON EXPLAIN、逐query CSV、统计快照、analyze成本；专属PG服务已停止。
- `cpu/{run}/`：24项CPU逐query CSV和summary，4800条。
- `audit.json`：独立标准库全量重算；`summary_metrics.csv`51行、`paired_estimates.csv`5100行、`metrics_configuration_ranges.csv`21行。
- 重建：`python Naru_Lab_20261109/scripts/joint_zero_audit.py`；不重跑PG或变更冻结查询。

## 保留失败与恢复

首次CPU summary中numpy bool比较求和得到numpy.int64，JSON序列化失败；发生在第一项CPU CSV完整写入后，PG已全部完成。修复为Python int计数，只恢复24项CPU。`failure_cpu_serialization_1/`保留原FAILED状态、traceback和首项原始CSV；当前`status.json`记录cpu_resume_only=True，不重复PG，不改600条冻结查询。根traceback.txt仍是保留历史失败，不覆盖当前COMPLETE状态。

后端操作回执在evidence/joint_zero_*；不占GPU、不修改共享系统或Python。
