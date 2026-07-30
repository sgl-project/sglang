#!/bin/bash
# Waits for the profile matrix to finish, then repeats the bench matrix into a second
# root so run-to-run noise can be quantified.
set -x

ROOT=${ROOT:-/scratch/sumpad_th3}
REPEAT_ROOT=${REPEAT_ROOT:-/scratch/sumpad_th3_rep}

for i in $(seq 1 200); do
  COUNT=$(ls "$ROOT"/profile/*/DONE 2>/dev/null | wc -l)
  if [ "$COUNT" -ge 8 ]; then
    break
  fi
  sleep 30
done

bash scratch_sumpad_th3/run_matrix.sh bench "$REPEAT_ROOT"
echo "REPEAT_ALL_DONE"
