#!/bin/bash
# The whole first profile matrix used --profile-by-stage, which with osl=1 leaves the
# profiler armed and dumps traces only on ranks that ran an extend forward. Wait until
# the repeat bench matrix is done (it owns the GPUs), then redo every config with a
# fixed step count instead.
set -x

ROOT=${ROOT:-/scratch/sumpad_th3}
REPEAT_ROOT=${REPEAT_ROOT:-/scratch/sumpad_th3_rep}
BAD=${BAD:-"default-breakable default-disabled no10414-breakable no10414-disabled sum-breakable sum-disabled max-breakable max-disabled"}

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
