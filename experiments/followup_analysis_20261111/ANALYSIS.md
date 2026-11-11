# 既有实验的四项离线分析

本分析只读取已冻结的本地结果，不训练模型、不访问GPU、不改原始文件，也未制作最终图表。重建命令：

```powershell
python Naru_Lab_20261109/scripts/followup_analysis_run.py
```

所有表均在本目录；audit.json记录输入路径、读取字节数、重算检查与行数。首次运行会流式扫描本地冻结DMV CSV，缓存精确full快照值频率；后续运行复用缓存。

**配对排名与稳定性。** 主负载每规模2000条查询记录，三个训练seed。查询条件按列排序后定义语义身份；重复条件先在每run内部取q-error中位数，再计算配对胜率，避免将相同谓词的重复MC估计计作独立查询。主表的原记录加权p95/p99保持不变，并在paired_rankings.csv中并列报告。

|规模|左方法 / 右方法|左方法胜率（三seed范围）|左方法p95较低的seed数|左方法p99较低的seed数|
|---|---|---:|---:|---:|
|1000000|own_made / own_resmade|46.8%–57.2%|3/3|0/3|
|1000000|own_made / official_resmade|53.5%–57.5%|3/3|2/3|
|1000000|own_resmade / official_resmade|47.2%–52.3%|1/3|2/3|
|5000000|own_made / own_resmade|56.5%–61.3%|3/3|1/3|
|5000000|own_made / official_resmade|58.9%–64.3%|3/3|2/3|
|5000000|own_resmade / official_resmade|50.7%–53.8%|3/3|2/3|
|full|own_made / own_resmade|53.3%–60.5%|3/3|2/3|
|full|own_made / official_resmade|57.3%–63.8%|3/3|2/3|
|full|own_resmade / official_resmade|52.2%–55.6%|3/3|2/3|

这里的胜率只描述保存查询上的误差排序，并不证明架构显著优越；MADE与ResMADE还同时存在masking/direct-io等功能差别。MADE及官方ResMADE主表只有MC seed0，无法从其三个训练seed中独立分离MC误差。paired_seed_consistency.csv给出同一语义条件在三个seed中持续胜出或发生反转的数量。

**MC与训练波动。** mc_spread_by_training_seed.csv保留每个固定checkpoint的三个MC重复范围；training_vs_mc_spread.csv比较三个checkpoint的MC均值差异和各checkpoint内部MC差异。以下均为描述性总体标准差，只有3×3网格，不是误差方差的严格因果分解或95%CI。

|规模|samples|p95训练均值之间SD|平均checkpoint内MC SD|语义query log(q-error)的MC SD中位数|
|---|---:|---:|---:|---:|
|1000000|100|0.0134|0.0112|0.0099|
|1000000|500|0.0163|0.0073|0.0049|
|1000000|2000|0.0248|0.0048|0.0024|
|1000000|8000|0.0236|0.0044|0.0012|
|5000000|100|0.0214|0.0206|0.0094|
|5000000|500|0.0106|0.0043|0.0047|
|5000000|2000|0.0060|0.0063|0.0025|
|5000000|8000|0.0048|0.0037|0.0012|
|full|100|0.0034|0.0119|0.0094|
|full|500|0.0097|0.0047|0.0046|
|full|2000|0.0060|0.0021|0.0025|
|full|8000|0.0076|0.0022|0.0014|

**尾部与失败模式。** tail_groups.csv按谓词数量、精确真值选择率、低基数(1–10行)、日期等值、Y标志及等值常量的full快照频率分组。稀有常量阈值在分析前固定为1e-5/1e-4/1e-3；它不等同于范围谓词选择率，且1M/5M使用的是full快照参照频率。每行保留记录数和语义去重数，小组的p95/p99不能视为稳健总体估计。

tail_cases_top10.csv固定选择每规模/方法三个seed中位q-error最高的10种语义，并同时保留三个seed估计、真值和原值谓词。它适合解释失败案例，不用来调参或宣称总体性能。单谓词/11谓词对照在predicate_extremes.csv，分别列正结果数量。

|规模|模型|最差语义的三seed中位q-error|真值|代表query ID|
|---|---|---:|---:|---|
|1000000|own_made|8.215|13|main_native_00669|
|1000000|own_resmade|5.917|23|main_native_01265|
|1000000|official_resmade|5.504|11|main_native_00483|
|5000000|own_made|27.359|59|main_native_00495|
|5000000|own_resmade|15.119|59|main_native_00495|
|5000000|official_resmade|10.410|59|main_native_00495|
|full|own_made|4.000|4|main_native_00453|
|full|own_resmade|4.000|4|main_native_00453|
|full|official_resmade|4.000|4|main_native_00453|

nontrivial_zero_summary.csv来自已有522条跨方法估计，按语义条件去重列出样本量和估计行数阈值。1M有14条、5M有4条查询记录，跨方法/seed重复不增加独立query数量。stress_v2的125条零结果全部存在空谓词域，只是语义一致性覆盖；这里不据此声称学会联合零结果。新联合零/近零实验由主线程另行负责，不混入旧结论。

**精度与成本。** accuracy_cost_by_run.csv把模型参数、checkpoint文件、采样行数据、PG逻辑catalog行字节分列；共同词典在模型/采样预算中同时排除。own/official ResMADE的相同预算与样本seed会重复同一份样本，不能当独立抽样。以下full主负载取各重复的p95及对应存储中位数，存储边界不同，因此只给并列表，不给物理存储Pareto排名。

|方法|p95中位数|报告的逻辑synopsis字节中位数|成本口径|
|---|---:|---:|---|
|own_made|1.1664|5750408|FP32 trainable parameters; common dictionaries excluded|
|own_resmade|1.1912|7814436|FP32 trainable parameters; common dictionaries excluded|
|official_resmade|1.1961|7814436|FP32 trainable parameters; common dictionaries excluded|
|uniform_sample|393.2500|440000|10000 materialized int32 rows, common dictionaries excluded|
|uniform_matched_own_resmade|2.0000|7814400|materialized int32 row bytes; common dictionaries excluded|
|PG-default|28.0318|6992|pg_column_size attributed catalog rows; not physical pages/indexes/WAL/heap/executables|
|PG-target1000|26.5738|16846|pg_column_size attributed catalog rows; not physical pages/indexes/WAL/heap/executables|
|PG-extstats|2.6162|24383|pg_column_size attributed catalog rows; not physical pages/indexes/WAL/heap/executables|

GPU列是共享未优化sampler的同步单query推断，含预建host mask到GPU的转换/传输，不含构造host numpy mask。PG列分别保存服务器规划和客户端往返，未执行SQL；不可计算二者的端到端SQL加速比。模型构建列用训练回执，PG用ANALYZE，采样/边际构建时间原实验未测量，留空而不补造。PG源代码构建、数据导入及共同数据表未并入统计构建成本。

**收敛、warmup与NLL。** convergence_by_run.csv保留33次正式拟合的每轮目标定义、epoch15/16到20的loss变化、末5轮线性趋势、warmup步数及占比。主矩阵20epochs在三个规模分别对应9780/48840/122480步，8000步warmup约占81.8%/16.4%/6.5%；规模效应仍混入优化预算。

|规模|模型|epoch16→20 loss下降（三seed均值）|末5轮斜率均值|warmup结束epoch|
|---|---|---:|---:|---:|
|1000000|own_made|0.03513|-0.00880|16.36|
|1000000|own_resmade|0.03448|-0.00826|16.36|
|1000000|official_resmade|0.03500|-0.00748|16.36|
|5000000|own_made|0.00561|-0.00140|3.28|
|5000000|own_resmade|0.00643|-0.00072|3.28|
|5000000|official_resmade|0.00303|-0.00159|3.28|
|full|own_made|0.00270|-0.00067|1.31|
|full|own_resmade|-0.00789|0.00134|1.31|
|full|official_resmade|0.00722|-0.00165|1.31|

1M末轮仍改善，缺少独立validation曲线，不能称充分收敛，也不能把训练loss下降解释为泛化改善。ResMADE每epoch loss含column masking，MADE不含；最终NLL是在同一训练表关闭masking计算，不能用它补成不存在的validation曲线。

full自写ResMADE末段masked loss略上行；随机mask训练目标和缺少独立validation限制了解释，单凭该段不能判断过拟合或数值发散。

nll_query_points.csv供后续散点图使用；1M ResMADE的9个拟合（3宽×3seed）中，NLL与查询分位数的Spearman仅作描述：
- qerror_p95：0.3667，n=9；width与seed同时变化，没有显著性/因果结论。
- qerror_p99：0.4667，n=9；width与seed同时变化，没有显著性/因果结论。

当前分析支持继续报告反直觉结果：更低NLL不保证每个查询分位数更低，更多MC采样也不保证每次p99单调下降。不需要为统一模型排名而重训。若报告要把规模收益归因于样本量，才需要单独固定优化步数；若要宣称MADE优于ResMADE，则需要更多推断重复或更谨慎的条件性表述。

原始输入入口：../main_evaluation_20261110/、../main_matrix_20261109/、../gpu_supplement_20261110/、../pg_supplement_20261110/、../matched_sampling_20261110/及数据快照workloads（项目根data/，完整路径见audit.json）。已有结果全部只读；本轮没有作图或生成report。
