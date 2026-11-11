# 补充查询与精确真值

本目录只补充查询和真值，不修改主实验的 `main/dev` 查询，也不包含训练或模型评估结果。运行状态以 `status.json` 的 `status` 字段为准。

三规模已于本次运行完成，共7200条查询及精确真值，CPU生成耗时492.68秒。全部原始JSONL、manifest、status和共享原值定义已同步本地，完整同步审计为PASS。为减少传输耗时，最终使用 `supplement_bundle.zip` 压缩包回传后解包；先前逐文件回传的部分文件保留并由同一冻结输出覆盖。

每个 `1000000/`、`5000000/`、`full/` 子目录均有：

| 文件 | 数量 | 固定生成规则 |
| --- | ---: | --- |
| `scale_shared.jsonl` | 1000 | 在 full 表随机行上产生5–11列查询，固定原值、列和操作符，再分别编译到三个规模的字典；保留零结果 |
| `predicate_sweep.jsonl` | 1100 | 1–11个谓词，每档100条；各规模独立锚定随机行，保留零结果 |
| `stress.jsonl` | 300 | 各100条保证零、显式NULL条件、单列边界条件 |

`shared_raw.jsonl` 是 shared 查询的原值定义。固定种子依次为20261120、20261121、20261122；stress按确定性循环生成，未读取模型误差。

每条查询包含 `query_id`、`workload`、`predicates`、`truth`、`selectivity`。每个谓词保留 `column_index`、`column`、`operator`、`value`、`value_code`、`allowed_codes`。模型评估应使用 `allowed_codes`；SQL评估应使用原值及操作符。

原值在较小规模的字典中缺失时，等值比较返回空mask，范围比较按原值的插入位置生成mask。日期按解析后的ISO时间比较，其他字段按字符串顺序比较。普通 `= NULL` 等比较不匹配任何行；`IS NULL` 和 `IS NOT NULL` 单独处理；空字符串保留为普通字符串值。

精确真值来自冻结编码表的mask交集，不依赖模型或PostgreSQL估计。独立SQLite小表验证包括77项原值/NULL/日期条件、25项双谓词交集，另有4项跨字典缺值/边界及4项非NULL字典/空字符串断言，回执位于 `../../evidence/supp_workload_test_extended.json`。本地同步审计 `../../evidence/supp_workload_audit.json` 检查全部行数、分组、字典范围、跨规模原值一致性和保证零条件。

stress是语义和边界的检查集，包含重复语义的case；`unique_semantic_queries` 和 `unique_by_group` 记录在同步审计中。报告时应披露重复，不将300行都视为独立随机性能样本。零结果需要独立报告，不能直接并入要求真值大于零的普通Q-error汇总。

每规模stress有62种不同的原值条件：保证零16种、NULL条件2种、边界44种。当前冻结数据只有一个字段的域含NULL，因此单列NULL条件只有 `IS NULL` / `IS NOT NULL` 两种。shared有998种不同原值查询；sweep在1M/5M/full分别有1030/1030/1027种，随机行锚定会自然重复低谓词数查询。

| 规模 | shared零结果 | sweep零结果 | stress零结果 |
| --- | ---: | ---: | ---: |
| 1M | 42 | 12 | 125 |
| 5M | 32 | 15 | 125 |
| full | 28 | 14 | 125 |

远程CPU复现入口：`scripts/supp_workload_build.py --snapshot data/snapshot_resume_20261109_172915 --output <新的独立目录>`。现有目录受保护，脚本拒绝覆盖。
