export CUDA_VISIBLE_DEVICES=4,5

#!/usr/bin/env bash
set -euo pipefail

TEST="test_piecewise_cuda_graph_22_gpu.py"
N=10

fail=0
for i in $(seq 1 "$N"); do
  echo "==================== Run $i / $N ===================="
  # -q reduces noise; remove if you want full unittest verbosity
  if python -m unittest -q "$TEST"; then
    echo "Run $i: PASS"
  else
    echo "Run $i: FAIL"
    fail=$((fail+1))
  fi
done

echo "====================================================="
echo "Done. total_failures=$fail / $N"
exit "$fail"
