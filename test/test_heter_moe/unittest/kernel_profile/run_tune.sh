#!/bin/bash
# Shard 25 (n_active, m_per_expert) cells of Task 2.prelim across GPUs 0..7.
# Round-robin assignment; each GPU does ceil(25/8)=4 cells sequentially, total
# wall time ~ 4 × ~2 min/cell = ~8-12 min wall (much less than the
# original tuning_fused_moe_triton.py because we tune per-cell at fixed shape).
set -euo pipefail
cd "$(dirname "$0")/../../../.."

OUT_DIR=test/test_heter_moe/unittest/kernel_profile/results
SHARD_DIR=$OUT_DIR/tune_shards
mkdir -p "$SHARD_DIR"

NS=(8 16 32 48 64)
MS=(32 64 128 256 512 1024 2048 4096)

# Build assignment list: round-robin over 8 GPUs.
gpu=0
for n in "${NS[@]}"; do
  for m in "${MS[@]}"; do
    echo "$gpu $n $m"
    gpu=$(( (gpu + 1) % 8 ))
  done
done > "$SHARD_DIR/assignment.txt"

PIDS=()
# For each GPU, run its assigned cells sequentially in one subshell.
for g in 0 1 2 3 4 5 6 7; do
  (
    while read -r gpu n m; do
      [[ "$gpu" == "$g" ]] || continue
      out="$SHARD_DIR/n${n}_bse${m}.json"
      log="$SHARD_DIR/n${n}_bse${m}.log"
      if [ -f "$out" ]; then
        echo "[gpu$g] skip n=$n bse=$m (exists)"
        continue
      fi
      echo "[gpu$g] start n=$n bse=$m → $out"
      CUDA_VISIBLE_DEVICES=$g conda run -n sglang --no-capture-output \
        python test/test_heter_moe/unittest/kernel_profile/tune_bf16_sparse.py \
        --n-active "$n" --m-per-expert "$m" --out "$out" \
        > "$log" 2>&1 || echo "[gpu$g] FAIL n=$n bse=$m (see $log)"
      echo "[gpu$g] done  n=$n bse=$m"
    done < "$SHARD_DIR/assignment.txt"
  ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"

# Merge shards into one JSON
python - << PYEOF
import glob, json, os
shard_dir = "$SHARD_DIR"
out = "$OUT_DIR/bf16_sparse_configs.json"
merged = {}
for path in sorted(glob.glob(os.path.join(shard_dir, "*.json"))):
    with open(path) as f:
        merged.update(json.load(f))
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"merged {len(merged)} cells → {out}")
PYEOF
