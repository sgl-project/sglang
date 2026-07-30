#!/bin/bash
# Waits for the bench matrix to finish (8 DONE markers), then runs evidence + profile matrices.
set -x

ROOT=${ROOT:-/scratch/sumpad_th3}

for i in $(seq 1 120); do
  COUNT=$(ls "$ROOT"/bench/*/DONE 2>/dev/null | wc -l)
  if [ "$COUNT" -ge 8 ]; then
    break
  fi
  sleep 30
done

bash scratch_sumpad_th3/run_matrix.sh evidence "$ROOT"
bash scratch_sumpad_th3/run_matrix.sh profile "$ROOT"
echo "REST_ALL_DONE"
