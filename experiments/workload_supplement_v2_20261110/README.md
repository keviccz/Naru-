# stress_v2：补全NULL组合

三规模各300条查询及真值已COMPLETE并同步本地，workload为 `stress_v2`，路径为 `<scale>/stress_v2.jsonl`。本地审计见 `../../evidence/supp_workload_v2_audit.json`（PASS）。shared与predicate sweep继续使用v1目录的对应文件。

v1的NULL循环使用 `j % 2` 同时选列和操作符，导致只覆盖 `fuel_type IS NULL` 与 `color IS NOT NULL`。v1 README中“只有一个字段域含NULL”的解释也错误；实际fuel_type和color均含NULL。全部v1数据与文档保留，本目录作为更正记录。

v2先显式枚举两列与两个操作符的笛卡尔积，再循环生成，四种条件各25条，共100条NULL检查。原100条保证零和100条边界查询的条件与真值逐条保持不变。生成和同步均有四组合覆盖断言，不再把列和操作符的选择耦合。每规模300行有64种不同原值条件（16保证零、4NULL、44边界），其中125行真值为零；这是含重复case的conformance检查集。

真值由冻结编码表精确计数。另用 `np.unique` 单独构建原值频数表，写入SQLite，并对四种NULL条件执行 `SUM(frequency)`；三规模12条SQL结果与真值全部相等。这是基于同一冻结数据的独立计数路径，并非另导入完整12.5M行SQLite表。PostgreSQL原始表的12条 `COUNT(*)` 由PG实验另留回执。

| 规模 | fuel_type NULL | fuel_type 非NULL | color NULL | color 非NULL |
| --- | ---: | ---: | ---: | ---: |
| 1M | 1251 | 998749 | 28532 | 971468 |
| 5M | 6295 | 4993705 | 143681 | 4856319 |
| full | 15500 | 12524979 | 360312 | 12180167 |

普通NULL比较与显式IS NULL仍严格区分；保证零组使用空allowed_codes，模型会在推断前直接返回零，不能把这组的正确性当作模型学会了不可能组合。真零单独报告；普通Q-error的真值大于零口径不得与其混合。
