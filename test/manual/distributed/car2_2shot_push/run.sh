#!/usr/bin/env bash
# usage: NP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh   (on the pod, from this dir)
set -euo pipefail
cd "$(dirname "$0")"
. /workspace/chunan/car2/env.sh
export RESULT="${RESULT:-sweep_tp${NP}.json}"
python3 -m torch.distributed.run --standalone --nproc-per-node="$NP" bench.py
