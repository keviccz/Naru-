#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
output="${1:?Usage: PYTHON_BIN=python3.8 [WHEELHOUSE=/path] bash run_cpu.sh /new/output}"
python_bin="${PYTHON_BIN:-python3.8}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,8), "This locked acceptance environment uses Python 3.8"'
if [ -e .venv ]; then echo 'Refusing to reuse an existing .venv; use a fresh copy of the portable package.' >&2; exit 2; fi
if [ -e "$output" ]; then echo 'Output already exists; preserve it and choose a new output path.' >&2; exit 2; fi
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p setup
"$python_bin" -m venv .venv >setup/venv.log 2>&1
if [ -n "${WHEELHOUSE:-}" ]; then
 .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements-cpu.txt >setup/install.log 2>&1
else
 .venv/bin/python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt >setup/install.log 2>&1
fi
.venv/bin/python -m pip check >setup/pip_check.txt 2>&1
.venv/bin/python -m pip freeze >setup/pip_freeze.txt
.venv/bin/python run_tiny.py --output "$output" >setup/pipeline.log 2>&1
.venv/bin/python run_tiny.py --output "$output" --summarize-only >setup/rebuild.log 2>&1
cat setup/pipeline.log setup/rebuild.log
