#!/bin/bash
# Shard Task 1 across GPUs 0..7. 9 n_cold values: GPUs 0..6 each take 1,
# GPU 7 takes 2.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

OUT_DIR=test/test_heter_moe/unittest/kernel_profile/results
mkdir -p "$OUT_DIR"

declare -A ASSIGN=(
  [0]="64"
  [1]="72"
  [2]="80"
  [3]="88"
  [4]="96"
  [5]="104"
  [6]="112"
  [7]="120 128"
)

PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
  for nc in ${ASSIGN[$gpu]}; do
    (
      echo "[gpu$gpu] start n_cold=$nc"
      CUDA_VISIBLE_DEVICES=$gpu conda run -n sglang --no-capture-output \
        python test/test_heter_moe/unittest/kernel_profile/bench_int4.py \
        --n-cold "$nc" \
        --out "$OUT_DIR/int4_table.csv" \
        > "$OUT_DIR/int4_n${nc}.log" 2>&1
      echo "[gpu$gpu] done n_cold=$nc"
    ) &
    PIDS+=($!)
  done
done
wait "${PIDS[@]}"

# Merge per-shard CSVs into one table (header from first, data from all).
TARGET="$OUT_DIR/int4_table.csv"
SHARDS=( $OUT_DIR/int4_table.n*.csv )
head -1 "${SHARDS[0]}" > "$TARGET"
for s in "${SHARDS[@]}"; do
  tail -n +2 "$s" >> "$TARGET"
done
echo "merged → $TARGET ($(wc -l < "$TARGET") lines)"
