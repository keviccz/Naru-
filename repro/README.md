# Tiny CPU reproduction

This portable package is an end-to-end code acceptance test. It does not measure DMV accuracy or GPU latency. It contains byte copies of the project's core implementation and the fixed upstream MADE, with source attribution in `SOURCE.json` and the upstream license beside its source.

Tested target: Linux x86_64, Python 3.8, PyTorch 2.4.1+cpu, NumPy 1.24.4. The PyTorch minor version matches the formal GPU environment, while the CPU build and tiny training rules differ. No system-site packages or user-site packages are permitted. The operating-system runtime and Python standard library remain host dependencies; this is not a fresh OS/container claim.

In a fresh extracted copy, run one command:

```bash
PYTHON_BIN=python3.8 bash run_cpu.sh /absolute/path/to/new_tiny_results
```

An official predownloaded wheelhouse can be used offline:

```bash
PYTHON_BIN=/path/to/python3.8 WHEELHOUSE=/path/to/packages bash run_cpu.sh /absolute/path/to/new_tiny_results
```

`run_cpu.sh` creates a new `.venv`, installs all pinned dependencies, runs `pip check`, executes the complete pipeline and independently rebuilds summary metrics from saved CSV. It rejects an existing environment or result directory. To repeat the acceptance run, extract a fresh package copy and choose a new output. The dependency lock is `requirements-cpu.txt`; installation uses only PyPI and the official PyTorch CPU index, or the supplied offline wheelhouse.

The raw 2,048-row CSV includes a deterministic region/vehicle dependence, ISO dates, SQL NULL represented by a distinct sentinel, and real empty strings. The pipeline saves dictionaries and encoded data, checks their raw roundtrip, and compares 14 fixed queries with SQLite COUNT. It covers ordinary NULL comparison, IS NULL/NOT NULL, wildcard positions, range boundaries, and an individually nonempty but jointly empty predicate combination.

Each of own MADE, own ResMADE and upstream ResMADE trains for 80 tiny epochs (1,280 steps, batch 128, width 32, depth 3, Adam lr0.003, seed0). It then saves and reloads its checkpoint, checks future-column independence, normalizes the enumerated joint distribution, and compares ancestral progressive sampling with exact model probabilities using ten fixed 1,024-sample repeats. Learned wildcard skipping is separately used for the saved tiny query estimates; it is not asserted equal to exact joint marginalization.

Outputs include environment paths/isolation checks, raw queries/truths, 240 epoch records, three checkpoints, 70 estimates from three models and two baselines, finite-loss/checkpoint/mask/probability checks, and summary/rebuilt-summary equality. These are functional checks, not a requirement to reproduce formal DMV loss or q-error. Full formal runs and checkpoints are preserved outside this package.

To rebuild only the metrics after a successful run:

```bash
.venv/bin/python run_tiny.py --output /path/to/completed_tiny_results --summarize-only
```
