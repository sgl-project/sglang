#!/bin/bash
# The first two profile configs ran before the stop_profile fix, so their skew
# traces are missing. Wait until the repeat bench matrix is done (it owns the GPUs),
# then redo just those configs.
set -x

ROOT=${ROOT:-/scratch/sumpad_th3}
REPEAT_ROOT=${REPEAT_ROOT:-/scratch/sumpad_th3_rep}
BAD=${BAD:-"default-breakable no10414-breakable"}

for i in $(seq 1 400); do
  COUNT=$(ls "$REPEAT_ROOT"/bench/*/DONE 2>/dev/null | wc -l)
  if [ "$COUNT" -ge 8 ]; then
    break
  fi
  sleep 30
done

INDEX=0
for name in $BAD; do
  VARIANT=${name%-*}
  GRAPH=${name##*-}
  OUT="$ROOT/profile/$name"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  PORT=$((31600 + INDEX)) bash scratch_sumpad_th3/run_one.sh "$VARIANT" "$GRAPH" profile "$OUT"
  touch "$OUT/DONE"
  INDEX=$((INDEX + 1))
done

echo "PROFILE_RERUN_ALL_DONE"
