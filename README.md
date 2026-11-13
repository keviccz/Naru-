# Naru++: Replicating and Extending Learned Cardinality Estimation

DSC5003 (CityU, Semester A 2026/27), Course Project Track 2: research paper replication.

Replicated paper: Zongheng Yang, Eric Liang, Amog Kamsetty, Chenggang Wu, Yan Duan, Xi Chen, Pieter Abbeel, Joseph M. Hellerstein, Sanjay Krishnan, Ion Stoica. *Deep Unsupervised Cardinality Estimation.* PVLDB 13(3): 279–292, 2019 (VLDB 2020). DOI: [10.14778/3368289.3368294](https://doi.org/10.14778/3368289.3368294).

Naru learns the joint distribution of a table with a deep autoregressive model (MADE / ResMADE) and estimates the selectivity of range and equality predicates with progressive sampling. This repository contains our own implementation of the core model and sampler, the frozen query workloads with exact ground truth, evaluation code, raw per-query results for the main experiment, and summary tables for all supplementary experiments.

## What is ours and what is not

| Path | Origin |
|---|---|
| `src/models.py` | **Our implementation**: masked autoregressive network (MADE and ResMADE with direct I/O and column masking), binary input encoding, predicate masks, progressive sampling with wildcard skipping; optional embedding inputs for large-domain columns (`embedding_columns`, used only by the improvement experiments) |
| `src/baselines.py` | **Ours**: exact-marginal independence estimator and uniform row sampling |
| `scripts/`, `tests/` | **Ours**: data freezing, encoding, workload generation, exact counts, training, evaluation, PostgreSQL baseline, audits and summaries |
| `vendor/naru/` | **Upstream code**, [naru-project/naru](https://github.com/naru-project/naru) at commit `264cf4e9c96c9e34422f9eebc455a714aeef0b57`, Apache License 2.0 (see `vendor/naru/LICENSE`). Used only as the "official ResMADE" comparison and as a reference for the optimizer/learning-rate schedule |
| `repro/package_source/` | Byte copies of `src/` and the vendored upstream files, bundled so the tiny CPU package is self-contained (`repro/SOURCE.json`) |

## Repository layout

```text
src/                     core implementation (model, sampler, baselines)
tests/                   synthetic correctness checks (exact enumeration, mask dependence, NULL/range semantics, upstream alignment)
scripts/                 pipeline scripts (see "Pipeline" below)
repro/                   self-contained tiny CPU end-to-end reproduction (one command)
vendor/naru/             pinned upstream Naru code (reference + official baseline)
data/snapshot_resume_20261109_172915/
  manifest.json          frozen DMV snapshot identity (source, update time, row count)
  encoded/*_domains.json per-scale ordered value dictionaries
  workloads/<scale>/     dev (200) and main (2000) queries with exact counts
experiments/
  main_matrix_20261109/          27 training runs: configs, per-epoch metrics, training_curves.csv, training_summary.csv
  pilot_1m_own_resmade_seed0/    the 1M own-ResMADE seed-0 run of the main matrix
  main_evaluation_20261110/      27 evaluations, raw per_query.csv for every run + summary tables
  gpu_supplement_20261110/       sampling-budget sweep (3 scales x 3 train seeds x 3 MC seeds x 4 budgets)
  size_supplement_20261110/      model width 128 / 512 at 1M
  matched_sampling_20261110/     uniform samples with storage matched to model parameters
  pg_supplement_20261110/        PostgreSQL 17.6 estimator, 3 configurations x 3 ANALYZE repeats (summaries only)
  workload_supplement*_2026111*/ shared, predicate-sweep and stress_v2 workloads
  joint_zero_*_20261111/         joint-zero / near-positive workload and all-method comparison
  supplement_analysis_20261110/, followup_analysis_20261111/   derived analysis tables
evidence/                audit receipts used by the summary scripts
```

## Data

New York State *Vehicle, Snowmobile, and Boat Registrations* ([data.ny.gov, dataset w4pv-hbkt](https://data.ny.gov/Transportation/Vehicle-Snowmobile-and-Boat-Registrations/w4pv-hbkt)), the same source as the paper. The snapshot linked by the original repository is no longer accessible, so we froze the current public version (source last updated 2026-11-03 UTC; 12,540,479 rows; the paper's 11 columns; duplicate rows kept; VIN not downloaded). Nested subsets of 1M and 5M rows and the full table are evaluated. The raw CSV (750 MB) is not stored here; `scripts/freeze_dmv.py` downloads it and `manifest.json` records its identity.

## Environment

- Formal runs: Linux, one NVIDIA RTX 3090 (24 GB), Python 3.8, PyTorch 2.4.1+cu121, NumPy, pandas.
- PostgreSQL baseline: PostgreSQL 17.6 built from source.
- Summary/audit scripts (`summarize_*.py`, `audit_*.py`, `followup_analysis_run.py`) use only the Python standard library.
- Tiny reproduction: Python 3.8 + PyTorch 2.4.1 CPU, pinned in `repro/requirements-cpu.txt`.

## Quick start

**1. Tiny end-to-end reproduction (CPU, a few minutes).** Creates a private venv, trains own MADE, own ResMADE and upstream ResMADE on a 2,048-row synthetic table, checks against SQLite counts and exact enumeration, and rebuilds the summary from CSV:

```bash
cd repro
PYTHON_BIN=python3.8 bash run_cpu.sh /absolute/path/to/new_output_dir
```

**2. Rebuild the main result tables from the raw per-query files in this repository (no GPU, standard library only):**

```bash
python scripts/summarize_evaluation.py
```

This regenerates `experiments/main_evaluation_20261110/{metrics_by_seed,metrics_seed_ranges,paired_resmade,grouped_metrics,worst_queries}.csv` and `evidence/main_evaluation_audit.json`.

**3. Correctness checks (needs PyTorch):**

```bash
python tests/check_models.py --device cpu     # exact enumeration, mask dependence, sampling convergence
python tests/check_baselines.py
python tests/check_training.py
python tests/check_workload.py
```

## Pipeline (full DMV runs)

| Step | Script |
|---|---|
| Freeze snapshot | `scripts/freeze_dmv.py` |
| Encode (ordered dictionaries, NULL as its own category) and build nested subsets | `scripts/encode_snapshot.py <snapshot_dir>`, `scripts/validate_real_encoding.py` |
| Main/dev workloads with exact counts | `scripts/build_workload.py <snapshot_dir> --scale {1000000,5000000,full} --output ...` |
| Supplementary workloads | `scripts/supp_workload_build.py`, `scripts/supp_workload_v2_build.py`, `scripts/joint_zero_backend.py` |
| Train | `scripts/train_model.py --encoded-dir ... --scale ... --model {own_made,own_resmade,official_resmade} --seed ... --run-dir ...` |
| Evaluate | `scripts/evaluate_matrix_model.py --run ... --queries ... --scope main_native --output ...`, `scripts/evaluate_supplement.py` |
| PostgreSQL baseline | `scripts/pg_build.py`, `scripts/pg_benchmark.py` |
| Summaries and audits | `scripts/summarize_*.py`, `scripts/audit_*.py`, `scripts/followup_analysis_run.py` |
| Improvement ablation | `scripts/train_improvement.py`, `scripts/evaluate_improvement.py`, `scripts/run_improvement_matched.py`, `scripts/analyze_improvements.py --experiment-root experiments/improvement_20261111` |

`scripts/run_*.py` are the batch drivers we used on the training server. They contain that server's absolute paths and interpreter location and are kept as a record of how each experiment was launched. Adjust the paths before reusing them.

## Main settings

Width 256, 5 layers, batch 2048, 20 epochs, warm-up 8000 steps, Adam with the upstream learning-rate schedule, natural column order, float32, seeds 0/1/2, no early stopping and no best-seed/epoch selection. Inference uses 2,000 progressive samples (MC seed 0); ResMADE uses wildcard skipping. The main workload follows the official generator: each query is anchored on a random row, filters 5–11 columns, and uses range predicates only on columns with at least 10 distinct values.

q-error = max(max(1,true)/max(1,est), max(1,est)/max(1,true)). For each seed we compute quantiles over the 2,000 queries, then report the median (and min–max) across the three seeds.

## Headline result (main workload, median across 3 training seeds)

| Scale | Method | median | p95 | p99 |
|---|---|---:|---:|---:|
| 1M | own MADE | 1.046 | 1.399 | 2.294 |
| 1M | own ResMADE | 1.054 | 1.463 | 2.197 |
| 1M | official ResMADE | 1.060 | 1.462 | 2.330 |
| 1M | exact-marginal independence | 1.243 | 17.00 | 294.7 |
| 1M | uniform sample (10k rows) | 1.063 | 28.05 | 124.0 |
| full | own MADE | 1.017 | 1.166 | 1.622 |
| full | own ResMADE | 1.021 | 1.191 | 1.622 |
| full | official ResMADE | 1.023 | 1.196 | 1.734 |
| full | exact-marginal independence | 1.226 | 26.46 | 581.0 |
| full | uniform sample (10k rows) | 1.066 | 393.3 | 1476 |

All learned models cut the tail error by one to three orders of magnitude relative to the independence and sampling baselines. Our ResMADE matches the official one query-by-query (median per-query error ratio 0.998–1.001). The simpler MADE is slightly better on most median/p95 cells, so we do not claim that ResMADE always wins. These numbers come from our snapshot and hardware and are not directly comparable to the paper's tables.

## Status

- Replication, baselines, PostgreSQL comparison and supplementary experiments: complete.
- Model improvements: complete. Fixed ablation on our ResMADE (mutual-information column order, 122,480 training steps at every scale, 32-dim embedding inputs for columns with domain > 64, and all three combined), 3 scales × 3 training seeds. We call the reproduced baseline **Naru** and the model with all three changes **Naru+**. Main-workload p95, median across seeds:

  | Variant | 1M | 5M | full |
  |---|---:|---:|---:|
  | Naru (baseline) | 1.463 | 1.295 | 1.191 |
  | Naru + order | 1.465 | 1.288 | 1.195 |
  | Naru + steps | 1.265 | 1.274 | 1.191 (= Naru) |
  | Naru + embed | 1.478 | 1.280 | 1.176 |
  | **Naru+** | 1.281 | 1.237 | 1.185 |

  Most of the gain comes from training longer at small scale (the 1M Naru baseline was under-trained). No variant, Naru+ included, wins on every metric: at full scale, Naru + embed and Naru+ raise p99 for two of three seeds and worsen the stress workload's positive-query p95 by 7–12%. Details, costs and negative results: `experiments/improvement_20261111/FINAL_ANALYSIS.md` (Chinese) and `docs/RESULTS_zh.md` §4; figures `fig12`–`fig17`.
- Figures: `figures/` (PNG + PDF), rebuilt by `python analysis/make_figures.py` (matplotlib, numpy, pandas); plotted numbers in `analysis/tables/`. Chinese result notes for the team: `docs/RESULTS_zh.md`.

## References

1. Z. Yang et al. Deep Unsupervised Cardinality Estimation. PVLDB 13(3), 2019.
2. naru-project/naru, https://github.com/naru-project/naru (commit 264cf4e).
3. M. Germain, K. Gregor, I. Murray, H. Larochelle. MADE: Masked Autoencoder for Distribution Estimation. ICML 2015.
4. New York State DMV, Vehicle, Snowmobile, and Boat Registrations, https://data.ny.gov/.
