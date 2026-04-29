#!/bin/bash
# Shard Task 3 measure-all-x across GPUs 0..7. Each M_global runs all 9 x
# candidates with actual paired-kernel measurement (using pinned autotuned
# BF16 tile). 24 M values across 8 GPUs ≈ 3 M values per GPU, ~30s/x × 9 ×
# 3 ≈ 13 min wall.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

OUT_DIR=test/test_heter_moe/unittest/kernel_profile/results
mkdir -p "$OUT_DIR"

M_VALUES=(32 64 96 128 256 512 1024 1536 2048 3072 4096)

PIDS=()
gpu=0
for m in "${M_VALUES[@]}"; do
  (
    g=$gpu
    echo "[gpu$g] start M=$m"
    CUDA_VISIBLE_DEVICES=$g conda run -n sglang --no-capture-output \
      python test/test_heter_moe/unittest/kernel_profile/compose_optimal.py \
      --measure-all-x --m-global "$m" \
      --out "$OUT_DIR/optimal_assignment.csv" \
      > "$OUT_DIR/task3_M${m}.log" 2>&1
    echo "[gpu$g] done  M=$m"
  ) &
  PIDS+=($!)
  gpu=$(( (gpu + 1) % 8 ))
done
wait "${PIDS[@]}"

# Merge per-M shards into one optimal_assignment.csv and one all_x.csv
python - << PYEOF
import csv, glob
out = "$OUT_DIR/optimal_assignment.csv"
shards = sorted(glob.glob("$OUT_DIR/optimal_assignment.M*.csv"))
header = None
rows = []
for s in shards:
    with open(s) as f:
        r = csv.reader(f)
        h = next(r)
        if header is None:
            header = h
        rows.extend(list(r))
rows.sort(key=lambda r: int(r[0]))
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows)
print(f"merged → {out} ({len(rows)} rows)")

ax = "$OUT_DIR/optimal_assignment.all_x.csv"
shards = sorted(glob.glob("$OUT_DIR/optimal_assignment.M*.all_x.csv"))
header = None; rows = []
for s in shards:
    with open(s) as f:
        r = csv.reader(f); h = next(r)
        if header is None: header = h
        rows.extend(list(r))
rows.sort(key=lambda r: (int(r[0]), int(r[1])))
with open(ax, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows)
print(f"merged → {ax} ({len(rows)} rows)")
PYEOF
