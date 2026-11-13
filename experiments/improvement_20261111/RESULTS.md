# ResMADE 固定消融实验

本地更新：2026-11-12T23:58:49.308937+08:00。

远程队列完成，本地原始CSV重算审计通过。

状态：`COMPLETE`；新拟合 33/33，复用 3/3，逻辑评估 720/720。

全部预设后端实验已完成。本地与远程审计均通过，当前无待执行训练或推断。

已确认的范围见 [DESIGN.md](DESIGN.md)，明确单元见 [plan.json](plan.json)。仅服务器物理GPU0；旧数据、结果和checkpoint保留。

正式训练每项固定最终step；epoch_metrics.jsonl保留loss、LR、吞吐量与资源。steps/all每6124步记录，真实epoch与见行数同时保留；不将虚拟窗口解释为同样的数据遍数。

已完成主负载的p95比值（变体/B；小于1为误差降低，范围仅描述训练seed）：

| 变体 | 规模 | 完成seed数 | p95比值中位数 | 范围 |
|---|---|---:|---:|---:|
| order | 1000000 | 3 | 0.9973 | 0.9858–1.0010 |
| order | 5000000 | 3 | 0.9879 | 0.9836–0.9983 |
| order | full | 3 | 1.0099 | 0.9964–1.0127 |
| steps | 1000000 | 3 | 0.8644 | 0.8387–0.8759 |
| steps | 5000000 | 3 | 0.9758 | 0.9653–0.9877 |
| steps | full | 3 | 1.0000 | 1.0000–1.0000 |
| embed | 1000000 | 3 | 0.9994 | 0.9947–1.0222 |
| embed | 5000000 | 3 | 0.9923 | 0.9770–0.9975 |
| embed | full | 3 | 0.9900 | 0.9870–0.9922 |
| all | 1000000 | 3 | 0.8861 | 0.8503–0.8864 |
| all | 5000000 | 3 | 0.9571 | 0.9363–0.9591 |
| all | full | 3 | 0.9944 | 0.9937–0.9958 |

已完成的 36 个主负载cell中，4 个p95高于B。保留负结果，不据此改预算、换seed或追加调参。

作图/分析入口：

- [训练曲线](analysis/training_curves.csv)、[训练成本与NLL](analysis/training_summary.csv)
- [逐seed指标](analysis/metrics_by_seed.csv)、[训练seed范围](analysis/metrics_seed_ranges.csv)、[MC范围](analysis/metrics_mc_ranges.csv)
- [B配对比较](analysis/paired_vs_baseline.csv)、[最差查询](analysis/worst_queries.csv)、[联合零pair](analysis/joint_pairs.csv)
- [本地审计](analysis/audit.json)、[队列状态](queue_status.json)、[监控状态](watch_status.json)

上述表在本地每次完整同步后重算；未闭合cell不会被当成完整结果。主分位数按冻结query记录口径，配对按语义去重。alias、B复用与共享matched抽样不作为独立重复。MC/训练seed的min–max不是置信区间。

全部config、loss、原始CSV、日志及失败证据在本地镜像；完整checkpoint和编码NPY保留远程。已有主结果参与形成改进动机，不能称为未见测试集的确认性验证。图表与report仍由后续Agent完成。

最终规模：635次实际新推断/899200条模型估计，另9份唯一匹配采样/18000条估计。720逻辑评估中含基线复用与预算别名；720曲线点中660点来自33新fit、60点来自3复用。

详细结论、负结果和推荐图表见 [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)；实时核验与33份新checkpoint检查见 [delivery_recovery_status.json](delivery_recovery_status.json)。GPU0最终1MiB/0%、无计算进程。早期失败已归档到attempts，收尾只补已有ID清单并重跑元数据审计，没有重复训练或推断。
