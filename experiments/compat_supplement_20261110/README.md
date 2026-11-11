# 官方推断独立验证与 PyTorch 1.13 兼容实验

本目录补足主实验采用共享采样器后留下的两项证据缺口。正式主实验的训练结果、查询结果和配置均不在这里重写。

## 预定方法

- 固定上游 Naru commit `264cf4e9c96c9e34422f9eebc455a714aeef0b57`。直接导入其未修改的 `estimators.ProgressiveSampling`，调用原始 `_sample_n`，与项目共享 `progressive_sample` 比较选择率。上游构造函数的缓存优化仍正常执行。
- 合成表为 4 列，域大小 `[3,4,3,2]`，固定随机权重。查询包括等值、范围、开头/中间 wildcard、NULL 比较、全 wildcard 和单过滤。对受支持的条件，同时穷举模型联合分布作为第三种独立核对。关闭和打开 wildcard skipping 分别运行。
- 真实验证固定为 1M 官方 ResMADE seed0 checkpoint 和开发集文件的前 12 个查询，不依结果选查询。主实验的 saved `allowed_codes` 必须逐项等于上游 OPS 生成的 mask。
- 每种采样器使用 1024 samples、5 个固定 seed。两者的随机数接口不同，不要求随机路径逐位相同。
- MC 容差预先固定为 `max(1e-6, 5%*max(|mean_upstream|,|mean_shared|), 6*sqrt(var_upstream/5+var_shared/5))`。原始重复均保存在 CSV，可重新计算 SE 与差异。该验证用于发现实现差异，不作为低选择率精度比较。
- 相同官方 checkpoint 的前 64 个真实行输出 logits、NLL，并对交替列设为 learned unknown 的输入输出 masked logits。PyTorch 2.4.1 与 1.13.1 的确定性 CPU 最大绝对差容差固定为 `2e-5`。采样结果跨版本沿用上述 MC 容差。
- 两个版本各运行 3 个 CPU Adam 步，batch128，启用官方 column masking，检查有限 loss 与参数更新。这仅证明短程训练 API 可运行。

训练 smoke 的 NumPy 随机 dropout 概率未冻结为相同状态；因此两个版本的这 3 步 loss 不能用于数值等价或训练性能对比。跨版本数值一致性的证据来自固定 checkpoint、相同 64 行输入的 eval logits/NLL、masked logits，以及独立重复的采样结果。

## 环境与边界

全部操作限制在 CPU；不使用或占用 GPU0。PyTorch 1.13.1 CPU wheel 从 PyTorch 官方 index 下载，传到项目目录，安装到私有 `venv_torch113`，使用 `--system-site-packages` 复用现有 NumPy/Pandas。不修改共享环境。

首次 1.13 启动时，继承的 Pandas 2.0.3 需要 `GLIBCXX_3.4.29`，而系统 `libstdc++` 较旧。兼容子进程通过 `LD_LIBRARY_PATH=/opt/conda/envs/R4.2/lib:...` 使用已安装的 conda C++ runtime；不替换系统库。失败 traceback 保留在 `torch113.log`，修复后版本导入回执在 `runtime113.log`。复现时应保留该进程环境变量。

远程初始仅有 `vendor/naru/made.py`。本目录 `upstream` 放置从本地固定 vendor 逐字复制的 `common.py`、`estimators.py`、`transformer.py`，仅补充导入依赖，不修改上游算法。

真实域传给上游的值采用保序数值代理，NULL 映射为 NaN。域顺序、类别编码和比较 mask 完全保持；采样器只需要域和 cardinality，不需要读取列数据。人工构造的轻量 `common.Table` 只服务于这个接口。

上游 shortcircuit 路径对少于两个约束列的查询会在 `masked_probs[1]` 触发 IndexError。本实验保留该异常并记录共享实现的输出，不对上游算法打补丁，也不将这些边界当作两实现一致性通过。NULL 普通比较的零概率仍需通过。IS NULL 属于独立压力测试，不冒充上游 API 支持。

## 文件

- `torch241/`、`torch113/`：各版本的 policy、environment、逐条件 checks、五次重复 CSV、边界异常、64行固定输出 NPZ、训练 smoke 与状态。
- `version_comparison.json`：两版本确定性和 MC 比较。
- `packages/source.json`：官方 wheel 来源、名称与大小；wheel 只作环境复现，不属于分析输入。
- `*.log`：完整标准输出和错误，首次依赖缺失启动也保留在 `torch241.log`。
- 本地 `evidence/compat_*.json`：远程启动、内核清理和上传回执。

本补充不能证明 2019 年原环境的完整训练可复现，也不证明上游 GPU 延迟。CPU 时长可用于验证执行成本，不能混入主实验 GPU 延迟图。

最终结论以实际 `summary.json` 和 `version_comparison.json` 为准。
