#!/bin/bash
# Shard Task 2.bench (8 n_hot values) across GPUs 0..7. One n_hot per GPU.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

OUT_DIR=test/test_heter_moe/unittest/kernel_profile/results
mkdir -p "$OUT_DIR"

PIDS=()
gpu=0
for nh in 8 16 24 32 40 48 56 64; do
  (
    echo "[gpu$gpu] start n_hot=$nh"
    CUDA_VISIBLE_DEVICES=$gpu conda run -n sglang --no-capture-output \
      python test/test_heter_moe/unittest/kernel_profile/bench_bf16.py \
      --n-hot "$nh" \
      > "$OUT_DIR/bf16_n${nh}.log" 2>&1
    echo "[gpu$gpu] done n_hot=$nh"
  ) &
  PIDS+=($!)
  gpu=$(( gpu + 1 ))
done
wait "${PIDS[@]}"

# Merge
TARGET=$OUT_DIR/bf16_table.csv
SHARDS=( $OUT_DIR/bf16_table.n*.csv )
head -1 "${SHARDS[0]}" > "$TARGET"
for s in "${SHARDS[@]}"; do
  tail -n +2 "$s" >> "$TARGET"
done
echo "merged → $TARGET ($(wc -l < "$TARGET") lines)"
