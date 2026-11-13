# ResMADE 改进：已确认实施方案

依据：[CODEX_HANDOFF_IMPROVEMENTS.md](../../CODEX_HANDOFF_IMPROVEMENTS.md)。2026-11-11用户已回答本轮两项决策；用户随后回复START，已确认共同理解并授权实施。

## 决策树

```text
自写 ResMADE 固定消融
├─ 验收：main_native p95为主要精度指标
│  └─ median/p99、NLL、参数与训练/推断成本并列，负结果保留
├─ 三规模 × 三训练seed × 四变体
│  ├─ V-order：训练表MI贪心列序，binary，20真实epochs
│  ├─ V-steps：自然列序，binary，122480步
│  ├─ V-embed：自然列序，混合输入编码，20真实epochs
│  └─ V-all：MI列序 + 混合编码 + 122480步
├─ embedding：本规模实际domain_size >64，维度32
│  ├─ 1M：reg_valid_date、color
│  └─ 5M/full：上述两列 + registration_class、state
├─ 新训练33次，V-steps/full三格复用B
└─ 正确性门槛 → GPU0固定队列 → 全范围冻结评估 → 本地审计/交接
```

这是对预先固定变体的消融验证，不以必须胜过B为验收条件，不根据中途主集成绩扩展调参、换seed或选epoch。已有主/补充负载结果曾用于形成改进动机，因此不声称全新未见测试集上的独立确认性验证。三个单因素加全开组合不是完整factorial，不独立估计全部交互项。

## 训练与实现

- 仅自写ResMADE；width256、layers5、batch2048、Adam及学习率公式、float32/确定性/TF32关闭等沿用B。MADE、官方ResMADE及旧结果保持冻结。
- 列序只用本规模完整训练NPY的经验两两MI，NULL作为类别；第一列按MI行和最大，之后按与已选集合MI之和最大，并列取原列下标小者。三seed共用一份MI/列序。原查询、值域及类别code不改，训练batch、domain列表、输出块和推断mask使用一致置换；配置同时保存置换/逆置换。
- embedding只改变输入；输出仍为各列类别softmax。embedding单元继承原列degree，MASK使用独立可学习unknown向量而非输出类别。固定采用上游`InitWeight`的normal(std=0.02)初始化，unknown零初始化，不作阈值/维度搜索。
- 现有binary默认行为须保持。比较冻结旧源码与新默认实现的state_dict/logits、随机数消费，并重跑原`tests/check_models.py`。
- V-steps的1M/5M从头训练。旧checkpoint无optimizer/RNG状态，不能用旧20epoch权重续训冒充同一优化轨迹。保留每真实epoch的完整shuffle和尾部短batch；1M为250整epoch+230批，5M为50整epoch+380批，full恰20epochs。
- 统一步数变体每6124步记录一次row-weighted虚拟epoch，共20点，同时保存真实epoch/step/实际见过行数。仅累计已有loss，不增加train-mode前向，不把虚拟epoch与B真实epoch解释成同一数据遍数。最终checkpoint固定在最后一步，无早停或择优。
- V-steps/full三seed复用B，显式标注`reused_from_baseline`；先验证原epoch循环与新step循环在1M seed0前100步的loss/状态一致，另检验489步附近真实epoch边界及tiny多轮截断。

## 正确性门槛

正式训练前，将原检查与四类新检查写入`evidence/improvement_checks.json`：

1. V-order/V-embed/V-all当前及未来列依赖掩码、直接embedding/MASK编码检查；tiny fixture显式触发embedding，不能只用小域默认binary。
2. 任意列序的数据/mask映射正确，相关小表上非skipping推断与模型穷举概率一致，精确oracle验证查询计数。
3. 全wildcard返回1；独立/精确可构造fixture验证skipping单列边际；学习到的相关分布skipping近似误差另报，不要求其必然精确等于联合模型边际。全wildcard直接返回不替代MASK编码检查。
4. 总步数、warmup、真实epoch尾批及最终截断精确；100步等价与跨epoch边界通过后才声明B/full复用成立。

## 评估与复用

模型输入、采样器和模型重建须适配新列序/混合编码，旧评估结果只读。新method为`own_resmade_<variant>`，逐query列保持主表格式。

| 范围 | 固定口径 |
|---|---|
| main2000、dev200、shared1000、sweep1100、stress_v2 300 | samples2000，MC0，随机种子100000+原查询索引 |
| shared共同正结果958条记录 | 直接从shared1000筛选，不重新编号、重推断；956种语义披露 |
| joint_zero_nearzero200 | 每训练seed的MC0/1/2；正例q-error、零估计行数/阈值、正例漏估、pair分离分别报告 |
| 采样预算 | 各变体3规模×3train×3MC×{100,500,2000,8000}；MC种子公式不变 |
| 新匹配预算CPU采样 | 仅参数字节变化的V-embed/V-all；沿用main2000，按唯一(scale,bytes,sample_seed)抽样并引用，共同字典两侧排除 |

main2000/MC0与budget2000/MC0共享同一次推断，显式记录来源；PG、既有CPU和B结果复用。当前本地B的dev仅1M seed0存在，实施前再查远程存量，仅补真正缺失的最多8份B dev推断，不重训或重跑已有。

33新checkpoint各19唯一推断格，共627格/897600条模型估计；若补8份B dev则635格/899200条。V-steps/full其余B评估均可复用。V-embed/V-all参数预算相同时匹配采样也共用，预计9份唯一抽样，不当独立重复。

主要结果每seed先算查询分位数，再取三seed中位数/min–max；配对按语义去重。MC重复单列，不把重复网格或seed范围称95%CI。延迟排除host numpy mask构建、包含GPU转换/传输及同步推断，与B边界一致。

## 资源、目录与交付

- 仅服务器物理GPU0 RTX3090，启动及各任务前查占用；忙则等待，不抢占其他项目。约10小时是训练估计，完整评估另需数小时，不以此作为未经实测的总耗时承诺。
- 开始实施时按实际日期新建`experiments/improvement_<date>/`，若存在则另起唯一后缀；本草案复制为该根`DESIGN.md`，保留最终确认记录。数据、旧checkpoints和历史结果不覆盖。
- 失败目录/日志保留，修复重试另起目录，成功与失败成本分别记录；不因负结果重试择优。没有自动超参搜索。
- 原33项正式训练仍是基线历史；本轮33项新训练另有variant标识。源代码和配置在队列开始前冻结；本地保存所有config、曲线与评估CSV，远程逐checkpoint核验。
- 分析CSV、配对、最差案例、参数量/耗时变化、负面结果与限制均写新根`RESULTS.md`；日志追加`BACKEND_LOG.md`，结束确认GPU0空闲。
- 交付期限沿用2026-11-23 23:59（Asia/Shanghai）。作图/中文报告由并行Agent负责。本轮只写Lab，不操作另一个Agent维护的`Naru_repo_20261111`或Git。
