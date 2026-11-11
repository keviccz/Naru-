# 兼容补充实验结果

受支持查询的独立上游推断验证完成。PyTorch 2.4.1 和 1.13.1 各有 24/24 个条件通过，包含 12 个合成条件和 12 条真实开发集查询。合成条件同时通过穷举模型概率核对。

相同 1M 官方 ResMADE seed0 checkpoint、相同 64 行输入：logits 最大绝对差 0，masked logits 0，NLL 1.90734863e-06，预定容差均为 2e-5。跨版本 MC 比较 48/48 通过。

上游原生边界另计：每个版本在 shortcircuit=True 的全 wildcard、单过滤查询各抛出一次 IndexError。这 2 个条件没有计入一致性通过数。完整 traceback 与共享实现输出保存在各版本 upstream_edge_tracebacks.json。主负载过滤数至少 5，不触发该分支。

两个版本均通过 3 个 CPU Adam 步的 column-masking 训练 API 检查。该结果只证明短程训练操作可用，不证明收敛、精度或正式 GPU 训练性能。

1.13 使用项目私有 CPU venv，共享环境保留 2.4.1+cu121。安装与源路径断言在 setup113.log。

分析入口：

- audit.json：计数、范围和最终核对。
- version_comparison.json：逐变量数值差、逐条件 MC 比较。
- torch241/ 与 torch113/ 下 *_repeats.csv：每个查询 5 次重复的原始选择率和 CPU 时长。
- checks.jsonl：均值、容差、穷举概率。
- fixed_checkpoint_outputs.npz：固定输入、logits、masked logits、NLL。
- training_smoke.json：3 步兼容性训练的原始 loss，勿混入正式训练曲线。

CPU 时长不得与主实验 GPU 延迟拼接；这项工作也不替代 2019 原环境的完整训练复现。
