#!/usr/bin/env bash
# usage: SGLANG_TREE=/path/to/sglang NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${SGLANG_TREE:?}/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export RESULT="${RESULT:-sweep_tp${NP}.json}"
python3 -m torch.distributed.run --standalone --nproc-per-node="$NP" bench.py
