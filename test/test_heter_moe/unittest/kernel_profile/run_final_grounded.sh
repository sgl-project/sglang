#!/bin/bash
# Drive the end-to-end final result generation:
#   1. Wait for tune_bf16_sparse_sep to finish (192 cells)
#   2. Merge tune_sep_shards/*.json -> bf16_sparse_configs_sep.json
#   3. Run Task 3 measure-all-x THREE times with cooldown between, save each
#      with a .runN suffix
#   4. Aggregate median across runs -> final x_star_curve.csv / .md
# Logs everything to results/grounded_final.log so progress can be polled
# without waking the parent.
set -uo pipefail
cd "$(dirname "$0")/../../../.."

OUT=test/test_heter_moe/unittest/kernel_profile/results
LOG=$OUT/grounded_final.log
> "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "WAIT: tune_bf16_sparse_sep procs"
while [[ $(pgrep -f tune_bf16_sparse_sep | wc -l) -gt 0 ]]; do
  sleep 30
done
log "tune done. shards=$(ls $OUT/tune_sep_shards/*.json | wc -l)"

log "MERGE shards into bf16_sparse_configs_sep.json"
python - << 'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json
shards = sorted(glob.glob("test/test_heter_moe/unittest/kernel_profile/results/tune_sep_shards/*.json"))
merged = {}
for s in shards:
    merged.update(json.load(open(s)))
out = "test/test_heter_moe/unittest/kernel_profile/results/bf16_sparse_configs_sep.json"
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"merged {len(merged)} cells -> {out}")
PYEOF

run_task3() {
    local suffix=$1
    log "TASK 3 run $suffix: launching shards"
    rm -f $OUT/optimal_assignment.M*.csv $OUT/optimal_assignment.M*.all_x.csv \
          $OUT/optimal_assignment.csv $OUT/optimal_assignment.all_x.csv

    bash test/test_heter_moe/unittest/kernel_profile/run_task3_full.sh \
         > "$OUT/task3_run${suffix}.log" 2>&1
    # Re-merge correctly (filter all_x out of optimal_assignment merge)
    python - << PYEOF 2>&1 | tee -a "$LOG"
import csv, glob
OUT="test/test_heter_moe/unittest/kernel_profile/results"
shards = sorted([s for s in glob.glob(f"{OUT}/optimal_assignment.M*.csv")
                 if not s.endswith(".all_x.csv")])
header=None; rows=[]
for s in shards:
    with open(s) as f:
        r=csv.reader(f); h=next(r)
        if header is None: header=h
        rows.extend(list(r))
rows.sort(key=lambda r: int(r[0]))
with open(f"{OUT}/optimal_assignment.run${suffix}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(header); w.writerows(rows)
print(f"run${suffix} optimal_assignment: {len(rows)} rows")

shards = sorted(glob.glob(f"{OUT}/optimal_assignment.M*.all_x.csv"))
header=None; rows=[]
for s in shards:
    with open(s) as f:
        r=csv.reader(f); h=next(r)
        if header is None: header=h
        rows.extend(list(r))
rows.sort(key=lambda r: (int(r[0]), int(r[1])))
with open(f"{OUT}/optimal_assignment.all_x.run${suffix}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(header); w.writerows(rows)
print(f"run${suffix} all_x: {len(rows)} rows")
PYEOF
    log "TASK 3 run $suffix: complete"
}

run_task3 1
log "COOLDOWN 90s"
sleep 90
run_task3 2
log "COOLDOWN 90s"
sleep 90
run_task3 3

log "AGGREGATE: median across 3 runs"
python - << 'PYEOF' 2>&1 | tee -a "$LOG"
import csv, statistics
OUT="test/test_heter_moe/unittest/kernel_profile/results"
all_data = {}
for run in (1, 2, 3):
    with open(f"{OUT}/optimal_assignment.all_x.run{run}.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            M = int(row[0]); x = int(row[1]); lat = float(row[2]); tile = row[3]
            all_data.setdefault((M, x), []).append((lat, tile))

# Median lat per (M, x); pick tile from the run whose lat is closest to median
median_rows = []
for (M, x), vals in sorted(all_data.items()):
    lats = [v[0] for v in vals]
    med = statistics.median(lats)
    closest = min(vals, key=lambda v: abs(v[0] - med))
    median_rows.append([M, x, f"{med:.4f}", closest[1],
                        f"{min(lats):.4f}", f"{max(lats):.4f}"])
with open(f"{OUT}/optimal_assignment.all_x.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["M_global","x","lat_ms","tile_key","lat_min_ms","lat_max_ms"])
    w.writerows(median_rows)
print(f"median across 3 runs: {len(median_rows)} (M,x) cells")

# Also write optimal_assignment.csv style table
import collections
by_M = collections.defaultdict(dict)
for r in median_rows:
    by_M[r[0]][r[1]] = float(r[2])
print("Per-M variance (max/min ratio across 3 runs):")
for (M, x), vals in sorted(all_data.items()):
    if x == 0:
        lats = [v[0] for v in vals]
        ratio = max(lats)/min(lats)
        print(f"  M={M:>5} x=0  lat=[{min(lats):.4f}..{max(lats):.4f}] ratio={ratio:.3f}")
PYEOF

log "MAKE REPORT (matrix)"
conda run -n sglang --no-capture-output \
    python test/test_heter_moe/unittest/kernel_profile/make_report.py 2>&1 | tee -a "$LOG"

log "DONE — final CSV at $OUT/x_star_curve.csv"
log "         markdown at $OUT/x_star_curve.md"
log "         all_x median at $OUT/optimal_assignment.all_x.csv"
