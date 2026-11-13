# PostgreSQL 实验完成与交接

状态：**COMPLETE，27 个统计重建单元、118,800 个原始 EXPLAIN 已完成并在本地逐条核对 PASS**。远程私有 PostgreSQL 服务已停止，cluster 保留用于复查。

数据为冻结 DMV 的 1M / 5M / full（12,540,479 行），每个单元分别评估 main_native 2000、scale_shared 1000、predicate_sweep 1100、stress_v2 300 条。主协议保持普通 NULL 比较语义；stress v1 保留在原负载目录，此处只评估覆盖完整四组合的 v2。

## 主实验结果

以下每个数是三次 ANALYZE 各自分位数的平均值，不是合并查询后的分位数。原始三次结果见 summary_metrics.csv。

|规模|配置|q-error median|p95|p99|平均 ANALYZE 秒|平均统计行字节|
|---|---|---:|---:|---:|---:|---:|
|1000000|PG-default|1.297|17.133|292.738|0.160|7024|
|1000000|PG-target1000|1.240|17.018|300.612|1.512|17039|
|1000000|PG-extstats|1.088|2.787|7.642|7.428|25186|
|5000000|PG-default|1.288|20.840|575.683|0.285|6971|
|5000000|PG-target1000|1.234|19.617|564.437|1.871|16882|
|5000000|PG-extstats|1.087|2.906|9.623|7.800|24578|
|full|PG-default|1.274|28.778|596.254|0.317|6976|
|full|PG-target1000|1.225|26.091|595.517|2.099|16872|
|full|PG-extstats|1.084|2.633|13.377|7.383|24419|

单列 target 从100提高到1000只带来有限改善；固定多列统计明显降低主查询尾部误差。这批负载的列相关结构能被小型多列统计有效捕捉。后续应与 Naru 的同查询结果直接对比；本文件没有据此检查或断言 Naru 的排名。

统计行字节是 pg_statistic / pg_statistic_ext_data 中属于本表的行之 pg_column_size 总和，记录的是服务器返回的 catalog row 大小，排除共享 catalog 索引与页面、WAL、可执行文件、数据表。单列/多列分别记录；数据表 pg_table_size 另列。不要将此列直接标成 PostgreSQL 总磁盘占用，或与包含不同边界的神经模型总内存混用。ndistinct 主要服务分组估计，不应归因到 WHERE 过滤收益。

## 证据与成本

- `../../evidence/pg_validation.json`：本地全量核验回执，27 cells / 118800 plans；计划行数与 CSV 估计逐条相同，配置与表名、q-error重算、行数和检查数量均通过。
- 各规模 `correctness.json`：实际表行数、8条开发查询 COUNT、11列 NULL 计数和普通 NULL 比较。
- 各规模 `supplement_correctness.json`：8条固定 stress_v2 查询的实际 COUNT。
- 各规模 `null_combo_correctness.json`：按实际 (column, operator) 枚举 fuel_type/color × IS NULL/IS NOT NULL 四组合；三规模共12条，全部与冻结真值相同。
- 各档 `repeat*/{scope}_plans.jsonl`：完整原始 SQL 和 JSON EXPLAIN；`{scope}_metrics.csv` 是逐查询估计/真值/延迟；`pg_stats*.json` 为该次重建后的实际统计。
- `summary_metrics.csv`：108行 scope-cell 汇总，便于后续作图；查询分布图应读取逐查询 CSV。
- 全部成功运行耗时229.7秒，包含三规模导入、独立 COUNT 与全部规划；27次 ANALYZE 合计86.6秒。该时间不包含首次依赖构建与启动故障排查。
- 本地原始输出包含351文件，原始约129.9MB，压缩包约11.5MB。源码与依赖输入另保留。

q_error 对真值零的行留空；q_error_floor1 对估计和真值分别下限设为1。Result 的精确零、扫描节点的 planner clamp=1、估计大于1分别保留。stress 的重复模式用于语义一致性检查，不应当作300个独立随机性能样本。三次 ANALYZE 是统计重建重复，不是神经训练 seed。server planning time 与客户端往返时间分开保存；CPU规划延迟和GPU推断延迟应说明设备与计时边界。

## 运行环境与保留事项

PostgreSQL17.6从官方源码项目内构建；缺少的bison/flex以及nss_wrapper仅解包到项目内。共享系统和Python没有安装改动。编译使用4线程，SQL禁用并行和JIT。
挂载根为UID763494/0700，该UID没有系统passwd记录；通过项目私有nss_wrapper映射为naru_pg，setpriv以该非root UID启动。只使用项目私有Unix socket，不监听TCP。连接身份、配置和停服记录见server_identity.json / server_config.txt / shutdown.log。
脚本在 ../../scripts/pg_*，远程根为 `<SERVER_ROOT>/experiments/pg_supplement_20261110`。失败回执和日志保留；早期失败均发生在正式统计评估前，成功结果未被替换。源码来源见source_metadata.json，详细方法和计量边界见README.md。

本交接只完成后端实验与证据整理；图表和report由后续Agent接手。
