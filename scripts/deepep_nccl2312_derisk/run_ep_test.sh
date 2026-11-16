#!/bin/bash
# Run on BOTH ranks (driven in parallel from the laptop).
# Usage: bash run_ep_test.sh <rank> <label> <repo-dir> [extra env as K=V,K=V]
set -uo pipefail
RANK=$1; LABEL=$2; REPO=$3; EXTRA=${4:-}
IFS=',' read -ra KVS <<< "$EXTRA"
for kv in "${KVS[@]}"; do [ -n "$kv" ] && export "$kv"; done

cd /workspace/derisk
{
  echo "=== $LABEL rank $RANK ==="
  date -u
  python3 -c "import deep_ep; print('deep_ep:', deep_ep.__file__)"
  python3 - <<'EOF'
import torch  # preloads the pip nvidia-nccl lib, like the real test processes
import ctypes
v = ctypes.c_int()
ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(v))
print("NCCL runtime version int (after torch import):", v.value)
EOF
  ls /sys/class/infiniband/ 2>/dev/null || true
  echo "EP_HYBRID_KERNEL=${EP_HYBRID_KERNEL:-<unset>}"
} 2>&1 | tee logs/${LABEL}_rank${RANK}.log

cd "$REPO"
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to the rank-0 node address} MASTER_PORT=8361 WORLD_SIZE=2 RANK=$RANK \
  timeout 3600 python3 tests/elastic/test_ep.py --num-processes 4 \
  2>&1 | tee -a /workspace/derisk/logs/${LABEL}_rank${RANK}.log
echo "EXIT=$? $LABEL rank $RANK" | tee -a /workspace/derisk/logs/${LABEL}_rank${RANK}.log
