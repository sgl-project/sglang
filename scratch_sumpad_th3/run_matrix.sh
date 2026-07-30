#!/bin/bash
# usage: run_matrix.sh <phase: bench|evidence|profile> [root]
set -x

PHASE=$1
ROOT=${2:-/scratch/sumpad_th3}
VARIANTS=${VARIANTS:-"default no10414 sum max"}
GRAPHS=${GRAPHS:-"breakable disabled"}

INDEX=0
for GRAPH in $GRAPHS; do
  for VARIANT in $VARIANTS; do
    INDEX=$((INDEX + 1))
    PORT=$((31500 + INDEX))
    OUT="$ROOT/$PHASE/$VARIANT-$GRAPH"
    if [ -f "$OUT/DONE" ]; then
      echo "SKIP $OUT"
      continue
    fi
    PORT=$PORT bash scratch_sumpad_th3/run_one.sh "$VARIANT" "$GRAPH" "$PHASE" "$OUT"
    echo "MATRIX_DONE phase=$PHASE variant=$VARIANT graph=$GRAPH"
    touch "$OUT/DONE"
  done
done
echo "MATRIX_ALL_DONE phase=$PHASE"
