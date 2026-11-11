# Tiny 干净环境端到端验收

状态：**COMPLETE，独立本地审计 PASS**。本次实际在远程 Linux 新建项目私有 CPU 环境，完成原始 CSV、编码、查询真值、三模型训练、checkpoint 重载、推断、基线与保存结果重建的完整链路。没有修改或重训正式 DMV 模型，没有使用 GPU。

## 实际完成的检查

- 2048 行合成 CSV，4 列，保留 SQL NULL 与真实空字符串的区别；编码与原值逐行 roundtrip。
- 14 条固定查询分别用 SQLite 原值 COUNT 和编码掩码计数；涵盖普通 NULL 比较、IS NULL/NOT NULL、范围边界、wildcard 与单列有匹配但联合为空的条件。本地审计再次独立 COUNT 全部 14 条。
- 自写 MADE、自写 ResMADE、固定上游 ResMADE 各训练 80 个 tiny epochs、1280 steps；有限 loss，最终无 mask NLL 均低于初始值。
- 三份 checkpoint 保存与重载后 logits 完全相同；未来列依赖掩码检查、联合概率归一化与渐进采样对模型穷举概率检查通过。每个模型/查询使用 10 个固定 1024-sample 重复，按预定 `max(0.015, 6*SE)` 容差比较；不要求训练模型等于原始数据真值概率。
- 共 84 项运行检查通过，240 条 tiny epoch 记录、3 个 checkpoint、70 行估计（14 查询 × 3 模型及 2 基线）已回传本地。240 条 tiny 训练记录独立保存，不能与正式训练的 660 条记录混用。
- 从保存的 CSV 独立重建全部 q-error/分位数，`summary.json` 与 `summary_rebuilt.json` 完全一致；`tiny_repro_audit.py` 再用标准库重算后 PASS。

## 环境证据

实际解释器为 Python 3.8.12，CPU PyTorch 2.4.1+cpu、NumPy 1.24.4。10 个运行依赖均精确锁定，安装只使用预先从 PyPI 与官方 PyTorch CPU 索引下载的 wheelhouse。bootstrap 工具为 pip 21.1.1、setuptools 56.0.0，来自该 Python 的 ensurepip，另列于 `audit.json`，没有将其假称跨主机统一锁定。

`portable/.venv/pyvenv.cfg` 为 `include-system-site-packages = false`，user site 被禁用。PyTorch、NumPy 文件路径与唯一 site-packages 均在新 venv 下；没有继承共享 site-packages、PYTHONPATH 或 LD_LIBRARY_PATH。`pip_check.txt` 为 `No broken requirements found.`。共享环境的 PyTorch 仍为 2.4.1+cu121，见 `shared_env_check.log`。

标准库和系统动态运行时仍使用原主机，因此这是干净 Python 运行环境验收，不是新 OS/container 验证，也不是历史 PyTorch 环境的完整训练复现。

## 可复制运行包

可移植源码在项目根的 `repro/`，独立 ZIP 为本目录 `portable.zip`。包内核心 `models.py`、`baselines.py`、上游 `made.py` 与当前项目文件逐字节相同；`SOURCE.json` 保留引用、上游 commit，LICENSE 随上游文件提供。`package_source_audit.json` 已核对 ZIP 与本地源码。

在 Linux x86_64 / Python 3.8 的新解包目录中执行：

```bash
PYTHON_BIN=python3.8 bash run_cpu.sh /absolute/path/to/new_tiny_results
```

离线使用本分支已有 wheelhouse：

```bash
PYTHON_BIN=/path/to/python3.8 WHEELHOUSE=/path/to/packages bash run_cpu.sh /absolute/path/to/new_tiny_results
```

命令会创建新 `.venv`，安装锁定依赖、检查依赖、执行完整 pipeline 并再次从 CSV 重建汇总；拒绝复用已有环境/结果目录。重跑时解包新目录并选择新输出，不覆盖原产物。

已完成结果的本地独立审计命令，不进行模型训练：

```powershell
python Naru_Lab_20261109/scripts/tiny_repro_audit.py
```

## 结果与边界

| 模型 | 初始无 mask NLL | 最终无 mask NLL | tiny 训练/检查耗时秒 |
|---|---:|---:|---:|
| 自写 MADE | 4.1873 | 2.6320 | 1.94 |
| 自写 ResMADE | 4.7982 | 2.6365 | 3.24 |
| 官方 ResMADE | 4.3739 | 2.6372 | 3.35 |

这些值只用于检查训练链路确实更新参数、checkpoint 与推断可用。它们不证明 DMV 准确率、GPU 延迟、充分收敛、架构排名或正式训练结果的跨机器可重复性。tiny 使用较小网络、固定学习率和不同训练轮数，其训练规则不等于正式矩阵。

主要证据：`status.json`、`audit.json`、`run1/environment.json`、`clean_environment_receipt.json`、`portable/setup/` 安装/检查日志、`run1/checks.json`、原始 CSV/JSON、checkpoint 和 `run1/summary_rebuilt.json`。依赖上传曾在146.8MB处中断，本次按既有远端字节断点续传48.1MB后一次运行成功；未重传整包或创建第二份训练。唯一独立审计脚本问题是 Windows 将远端 POSIX 路径转换为反斜杠，已修正为原始 POSIX 字符串前缀比较，不影响实验运行或产物。
