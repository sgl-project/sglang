# KV Transfer Mooncake Background Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the remaining KV Transfer experiment matrix for `2x100`, `4x50`, `2x200`, and `4x100` with foreground measurement rate limits and Mooncake-generated background traffic at `1/10/50/90Gbps` per active lane.

**Architecture:** Each run is named `${PROFILE}_bg${BG}_cap${LANE_CAP}_moonbg` under `/tmp/kv-transfer-bench/manual/`. Node `102` runs two Mooncake target sets per active lane: one foreground target and one background target. Node `099` runs the long-lived Mooncake background initiators first, then runs foreground measurement initiators against separate target sessions. Nodes `100` and `101` are not used in this new background-flow plan.

**Tech Stack:** SGLang/Mooncake `kv_transfer_latency.py`, Docker image `sglang-pd-switch:tianciJ`, RDMA IPv6 GID index 3, host shell, host Python CSV/JSONL aggregation, Markdown reports under `docs/superpowers/reports/`.

---

## Files

- Create: `docs/superpowers/plans/2026-06-24-kv-transfer-mooncake-background-matrix.md`
- Future create or modify: `docs/superpowers/reports/2026-06-24-kv-transfer-capped-background-report.md`
- Reference report style: `docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md`
- Remote script used by all Docker runs: `/root/kv_transfer_bench/kv_transfer_latency.py`
- Per-run remote output root: `/tmp/kv-transfer-bench/manual/${RUN}`

## Experiment Matrix

Run all four background levels for each profile.

| Profile | RUN prefix | Lanes | Shards | Foreground cap per lane | Background cap per lane | Target max bytes per lane | Shard size list |
|---|---|---:|---:|---:|---:|---:|---|
| `2x100` | `200_2x100` | `0 1` | 2 | `100Gbps` | `BG` | `1GB` | `DENSE_SHARD_SIZES_2` |
| `4x50` | `200_4x50` | `0 1 2 3` | 4 | `50Gbps` | `BG` | `512MB` | `DENSE_SHARD_SIZES_4` |
| `2x200` | `400_2x200` | `0 1` | 2 | `200Gbps` | `BG` | `1GB` | `DENSE_SHARD_SIZES_2` |
| `4x100` | `400_4x100` | `0 1 2 3` | 4 | `100Gbps` | `BG` | `512MB` | `DENSE_SHARD_SIZES_4` |

Background levels are direct Mooncake per-lane caps:

```text
BG=1   -> Mooncake background cap 1Gbps per active lane
BG=10  -> Mooncake background cap 10Gbps per active lane
BG=50  -> Mooncake background cap 50Gbps per active lane
BG=90  -> Mooncake background cap 90Gbps per active lane
```

Run order:

```text
200_4x50_bg1_cap50_moonbg
200_4x50_bg10_cap50_moonbg
200_4x50_bg50_cap50_moonbg
200_4x50_bg90_cap50_moonbg
400_2x200_bg1_cap200_moonbg
400_2x200_bg10_cap200_moonbg
400_2x200_bg50_cap200_moonbg
400_2x200_bg90_cap200_moonbg
400_4x100_bg1_cap100_moonbg
400_4x100_bg10_cap100_moonbg
400_4x100_bg50_cap100_moonbg
400_4x100_bg90_cap100_moonbg
```

If `200_2x100` needs to be rerun under this new background-flow plan, use:

```text
200_2x100_bg1_cap100_moonbg
200_2x100_bg10_cap100_moonbg
200_2x100_bg50_cap100_moonbg
200_2x100_bg90_cap100_moonbg
```

## Common Constants

```bash
# 099 source IPv6 addresses
SRC_IP_0=fd03:4514:80:6240::1
SRC_IP_1=fd03:4514:80:6241::1
SRC_IP_2=fd03:4514:80:6242::1
SRC_IP_3=fd03:4514:80:6243::1

# 102 target IPv6 addresses
TGT_IP_0=fd03:4514:80:5f00::1
TGT_IP_1=fd03:4514:80:5f01::1
TGT_IP_2=fd03:4514:80:5f02::1
TGT_IP_3=fd03:4514:80:5f03::1
```

Dense size lists:

```bash
export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB
```

Lane placement:

| Lane | 099 source IPv6 | 102 target IPv6 | RDMA device |
|---:|---|---|---|
| 0 | `fd03:4514:80:6240::1` | `fd03:4514:80:5f00::1` | `mlx5_bond_0` |
| 1 | `fd03:4514:80:6241::1` | `fd03:4514:80:5f01::1` | `mlx5_bond_1` |
| 2 | `fd03:4514:80:6242::1` | `fd03:4514:80:5f02::1` | `mlx5_bond_2` |
| 3 | `fd03:4514:80:6243::1` | `fd03:4514:80:5f03::1` | `mlx5_bond_3` |

## Task 1: Choose One Run

**Files:**
- Create remotely: `/tmp/kv-transfer-bench/manual/${RUN}/raw/`

- [ ] **Step 1.1: Choose the next run**

For the current next run, use this example:

```bash
PROFILE=200_4x50
BG=1
LANE_CAP=50
BG_RATE=1
LANES=(0 1 2 3)
SHARDS=4
MAX_BYTES=512MB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_4"
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}_moonbg
OUT=/tmp/kv-transfer-bench/manual/${RUN}
```

For other profiles, change only these variables:

| RUN | `PROFILE` | `BG` | `LANE_CAP` | `BG_RATE` | `LANES` | `SHARDS` | `MAX_BYTES` | `DENSE_SHARD_SIZES` |
|---|---|---:|---:|---:|---|---:|---:|---|
| `200_2x100_bg1_cap100_moonbg` | `200_2x100` | 1 | 100 | 1 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_2x100_bg10_cap100_moonbg` | `200_2x100` | 10 | 100 | 10 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_2x100_bg50_cap100_moonbg` | `200_2x100` | 50 | 100 | 50 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_2x100_bg90_cap100_moonbg` | `200_2x100` | 90 | 100 | 90 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_4x50_bg1_cap50_moonbg` | `200_4x50` | 1 | 50 | 1 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg10_cap50_moonbg` | `200_4x50` | 10 | 50 | 10 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg50_cap50_moonbg` | `200_4x50` | 50 | 50 | 50 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg90_cap50_moonbg` | `200_4x50` | 90 | 50 | 90 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_2x200_bg1_cap200_moonbg` | `400_2x200` | 1 | 200 | 1 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg10_cap200_moonbg` | `400_2x200` | 10 | 200 | 10 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg50_cap200_moonbg` | `400_2x200` | 50 | 200 | 50 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg90_cap200_moonbg` | `400_2x200` | 90 | 200 | 90 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_4x100_bg1_cap100_moonbg` | `400_4x100` | 1 | 100 | 1 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg10_cap100_moonbg` | `400_4x100` | 10 | 100 | 10 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg50_cap100_moonbg` | `400_4x100` | 50 | 100 | 50 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg90_cap100_moonbg` | `400_4x100` | 90 | 100 | 90 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |

## Task 2: Start Foreground And Background Targets On 102

**Files:**
- Create remotely: `$OUT/raw/target-bond${lane}.log`
- Create remotely: `$OUT/raw/target-bg-bond${lane}.log`
- Create remotely: `$OUT/raw/rdma-rcv-monitor.csv`

- [ ] **Step 2.1: Configure the run on 102**

Run on `lingjun-102`:

```bash
cd /root/kv_transfer_bench

export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB

PROFILE=200_4x50
BG=1
LANE_CAP=50
BG_RATE=1
LANES=(0 1 2 3)
SHARDS=4
MAX_BYTES=512MB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_4"
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}_moonbg
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT/raw"

declare -A TGT_IPS=(
  [0]=fd03:4514:80:5f00::1
  [1]=fd03:4514:80:5f01::1
  [2]=fd03:4514:80:5f02::1
  [3]=fd03:4514:80:5f03::1
)
```

- [ ] **Step 2.2: Clean only this run if it was already started**

Run on `lingjun-102`:

```bash
docker ps --format '{{.Names}}' | grep "^kv_${RUN}_" | xargs -r docker rm -f

if [ -s "$OUT/raw/rdma-monitor.pid" ]; then
  kill "$(cat "$OUT/raw/rdma-monitor.pid")" 2>/dev/null || true
fi

ss -ltnp | grep -E ':(18500|18501|18502|18503)\b' || true
```

Expected: no old container or monitor process from the same `RUN` remains. Ports `18500-18503` are not used by this Mooncake-background plan; if old `ib_write_bw` processes remain, kill them before continuing.

- [ ] **Step 2.3: Start foreground targets and background targets**

Run on `lingjun-102`:

```bash
for lane in "${LANES[@]}"; do
  nohup docker run --rm --name kv_${RUN}_target_bond${lane} \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma \
    -e IB_DEVICE=mlx5_bond_${lane} \
    -e MC_USE_IPV6=1 \
    -e MC_GID_INDEX=3 \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role target --host ${TGT_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --max-bytes ${MAX_BYTES} --target-info-file /tmp/kv-transfer-bench/target-bond${lane}.json" \
    > "$OUT/raw/target-bond${lane}.log" 2>&1 &

  nohup docker run --rm --name kv_${RUN}_target_bg_bond${lane} \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma \
    -e IB_DEVICE=mlx5_bond_${lane} \
    -e MC_USE_IPV6=1 \
    -e MC_GID_INDEX=3 \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role target --host ${TGT_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --max-bytes ${MAX_BYTES} --target-info-file /tmp/kv-transfer-bench/target-bg-bond${lane}.json" \
    > "$OUT/raw/target-bg-bond${lane}.log" 2>&1 &
done

for lane in "${LANES[@]}"; do
  until grep -q 'target_ready=true' "$OUT/raw/target-bond${lane}.log"; do sleep 1; done
  until grep -q 'target_ready=true' "$OUT/raw/target-bg-bond${lane}.log"; do sleep 1; done
done

grep '^TARGET_INFO_JSON=' "$OUT/raw"/target-bond*.log
grep '^TARGET_INFO_JSON=' "$OUT/raw"/target-bg-bond*.log
```

Expected: one foreground `TARGET_INFO_JSON` and one background `TARGET_INFO_JSON` per active lane.

- [ ] **Step 2.4: Start a 102 RDMA receive counter monitor**

Run on `lingjun-102`:

```bash
echo "ts,dev,rcv_Gbps" > "$OUT/raw/rdma-rcv-monitor.csv"
(
  declare -A last
  for lane in "${LANES[@]}"; do
    dev=mlx5_bond_${lane}
    last[$dev]=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
  done

  tlast=$(date +%s)
  while true; do
    sleep 2
    now=$(date +%s)
    ts=$(date -Iseconds)
    dt=$((now - tlast))

    for lane in "${LANES[@]}"; do
      dev=mlx5_bond_${lane}
      cur=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
      prev=${last[$dev]}
      gbps=$(awk -v cur="$cur" -v prev="$prev" -v dt="$dt" 'BEGIN{printf "%.3f",(cur-prev)*32/dt/1e9}')
      echo "$ts,$dev,$gbps"
      last[$dev]=$cur
    done

    tlast=$now
  done
) >> "$OUT/raw/rdma-rcv-monitor.csv" &
echo $! > "$OUT/raw/rdma-monitor.pid"
```

Expected:

```bash
sleep 5
tail -n 20 "$OUT/raw/rdma-rcv-monitor.csv"
```

The file should contain one row per active `mlx5_bond_N` every two seconds.

- [ ] **Step 2.5: Print target JSON exports for 099**

Run on `lingjun-102`:

```bash
for lane in "${LANES[@]}"; do
  json=$(sed -n 's/^TARGET_INFO_JSON=//p' "$OUT/raw/target-bond${lane}.log" | tail -1)
  echo "export TARGET_JSON_${lane}='$json'"
done

for lane in "${LANES[@]}"; do
  json=$(sed -n 's/^TARGET_INFO_JSON=//p' "$OUT/raw/target-bg-bond${lane}.log" | tail -1)
  echo "export TARGET_BG_JSON_${lane}='$json'"
done
```

Copy the printed `export TARGET_JSON_${lane}=...` and `export TARGET_BG_JSON_${lane}=...` lines to `lingjun-099`. Do not execute them on `102`.

## Task 3: Start Mooncake Background Initiators On 099

**Files:**
- Create remotely on `099`: `$OUT/raw/bgmoon-init-bond${lane}.log`
- Create remotely on `099`: `$OUT/raw/bg-mooncake-pids.txt`
- Create remotely on `099`: `$OUT/raw/bgmoon-bond${lane}-summary.csv`
- Create remotely on `099`: `$OUT/raw/bgmoon-bond${lane}-samples.jsonl`

- [ ] **Step 3.1: Configure the run on 099**

Run on `lingjun-099`:

```bash
cd /root/kv_transfer_bench

export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB

PROFILE=200_4x50
BG=1
LANE_CAP=50
BG_RATE=1
LANES=(0 1 2 3)
SHARDS=4
MAX_BYTES=512MB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_4"
CHUNK_SIZE=64MB
BG_DURATION=1200
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}_moonbg
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT/raw"

declare -A SRC_IPS=(
  [0]=fd03:4514:80:6240::1
  [1]=fd03:4514:80:6241::1
  [2]=fd03:4514:80:6242::1
  [3]=fd03:4514:80:6243::1
)
```

Paste the exact `export TARGET_BG_JSON_${lane}=...` lines printed by `102` in Task 2.5.

- [ ] **Step 3.2: Launch Mooncake background initiators**

Run on `lingjun-099`:

```bash
: > "$OUT/raw/bg-mooncake-pids.txt"

for lane in "${LANES[@]}"; do
  var=TARGET_BG_JSON_${lane}
  target_json=${!var}

  nohup docker run --rm --name kv_${RUN}_bgmoon_init_bond${lane} \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma \
    -e IB_DEVICE=mlx5_bond_${lane} \
    -e MC_USE_IPV6=1 \
    -e MC_GID_INDEX=3 \
    -e TARGET_INFO_JSON="$target_json" \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role initiator --host ${SRC_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --sizes ${MAX_BYTES} --background-bytes ${MAX_BYTES} --background-duration-seconds ${BG_DURATION} --rate-limit-gbps ${BG_RATE} --chunk-size ${CHUNK_SIZE} --summary-csv /tmp/kv-transfer-bench/bgmoon-bond${lane}-summary.csv --samples-jsonl /tmp/kv-transfer-bench/bgmoon-bond${lane}-samples.jsonl" \
    > "$OUT/raw/bgmoon-init-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-mooncake-pids.txt"
done

sleep 10
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "kv_${RUN}_bgmoon" || true
grep -H -E 'summary_csv=|mode.*background|Traceback|RuntimeError|failed|ret=-1|Unable|ERROR' "$OUT/raw"/bgmoon-init-bond*.log || true
```

Expected: one running `kv_${RUN}_bgmoon_init_bond${lane}` container per active lane. Error grep should not show failures. On `102`, `rdma-rcv-monitor.csv` should rise near the configured background level before foreground starts.

## Task 4: Run Foreground Measurement On 099

**Files:**
- Create remotely: `$OUT/raw/init-bond${lane}-dense.log`
- Create remotely: `$OUT/raw/shard-bond${lane}-dense-summary.csv`
- Create remotely: `$OUT/raw/shard-bond${lane}-dense-samples.jsonl`

- [ ] **Step 4.1: Paste foreground target JSONs on 099**

Still on `lingjun-099`, paste the exact `export TARGET_JSON_${lane}=...` lines printed by `102` in Task 2.5.

- [ ] **Step 4.2: Launch foreground initiators**

Run on `lingjun-099`:

```bash
pids=()
for lane in "${LANES[@]}"; do
  var=TARGET_JSON_${lane}
  target_json=${!var}

  nohup docker run --rm --name kv_${RUN}_init_bond${lane}_dense \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma \
    -e IB_DEVICE=mlx5_bond_${lane} \
    -e MC_USE_IPV6=1 \
    -e MC_GID_INDEX=3 \
    -e TARGET_INFO_JSON="$target_json" \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role initiator --host ${SRC_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --sizes ${DENSE_SHARD_SIZES} --warmup 3 --repeat 20 --rate-limit-gbps ${LANE_CAP} --chunk-size ${CHUNK_SIZE} --summary-csv /tmp/kv-transfer-bench/shard-bond${lane}-dense-summary.csv --samples-jsonl /tmp/kv-transfer-bench/shard-bond${lane}-dense-samples.jsonl" \
    > "$OUT/raw/init-bond${lane}-dense.log" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

grep -H 'summary_csv=' "$OUT/raw"/init-bond*-dense.log
grep -H -E 'Traceback|RuntimeError|failed|ret=-1|Unable|ERROR' "$OUT/raw"/init-bond*-dense.log || true
```

Expected: one `summary_csv=` line per active lane and no error lines.

## Task 5: Aggregate Logical Transfer Results

**Files:**
- Create remotely on `099`: `$OUT/aggregated-summary.csv`

- [ ] **Step 5.1: Aggregate shards by logical size and iteration**

Run on `lingjun-099`:

```bash
python3 - "$OUT" "$SHARDS" <<'PY'
import csv, glob, json, math, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
shards = int(sys.argv[2])
raw = out / "raw"

def pct(vals, p):
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def human(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.2f}GiB"
    return f"{n / 1024**2:.2f}MiB"

groups = defaultdict(list)
errors = defaultdict(int)

for path in sorted(glob.glob(str(raw / "shard-bond*-dense-samples.jsonl"))):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            logical_bytes = int(row["bytes"]) * shards
            key = (logical_bytes, int(row["iteration"]))
            if int(row.get("ret", 0)) == 0:
                groups[key].append(float(row["latency_ms"]))
            else:
                errors[logical_bytes] += 1

by_size = defaultdict(list)
for (logical_bytes, _), vals in groups.items():
    if len(vals) == shards:
        by_size[logical_bytes].append(max(vals))
    else:
        errors[logical_bytes] += 1

summary = []
for logical_bytes in sorted(by_size):
    vals = by_size[logical_bytes]
    mean = sum(vals) / len(vals)
    p50 = pct(vals, 0.50)
    summary.append({
        "bytes": logical_bytes,
        "human_bytes": human(logical_bytes),
        "shard_count": shards,
        "repeat_count": len(vals),
        "error_count": errors[logical_bytes],
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p90": pct(vals, 0.90),
        "latency_ms_p99": pct(vals, 0.99),
        "latency_ms_min": min(vals),
        "latency_ms_max": max(vals),
        "bandwidth_GBps_p50": (logical_bytes / 1024**3) / (p50 / 1000),
        "bandwidth_GBps_mean": (logical_bytes / 1024**3) / (mean / 1000),
    })

if not summary:
    raise SystemExit("no complete shard groups found")

with open(out / "aggregated-summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("row_count", len(summary))
print("error_sum", sum(r["error_count"] for r in summary))
print(out / "aggregated-summary.csv")
PY

cat "$OUT/aggregated-summary.csv"
```

Expected:

```text
row_count 21
error_sum 0
aggregated-summary.csv contains logical bytes from 1MiB through 2GiB.
```

- [ ] **Step 5.2: Capture key rows**

Run on `lingjun-099`:

```bash
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
want = {"512.00MiB", "1.00GiB", "2.00GiB"}
with open(out / "aggregated-summary.csv") as f:
    for row in csv.DictReader(f):
        if row["human_bytes"] in want:
            print(
                row["human_bytes"],
                "p50", row["latency_ms_p50"],
                "p90", row["latency_ms_p90"],
                "p99", row["latency_ms_p99"],
                "bw_p50", row["bandwidth_GBps_p50"],
                "errors", row["error_count"],
            )
PY
```

Expected: exactly three rows, all with `errors 0`.

## Task 6: Stop This Run Cleanly

**Files:**
- Read remotely on `099`: `$OUT/raw/bg-mooncake-pids.txt`
- Read remotely on `102`: `$OUT/raw/rdma-monitor.pid`

- [ ] **Step 6.1: Stop Mooncake background initiators on 099**

Run on `lingjun-099`:

```bash
if [ -s "$OUT/raw/bg-mooncake-pids.txt" ]; then
  xargs -r kill < "$OUT/raw/bg-mooncake-pids.txt" 2>/dev/null || true
fi

docker ps --format '{{.Names}}' | grep "^kv_${RUN}_bgmoon_" | xargs -r docker rm -f
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "kv_${RUN}" || true
```

Expected: no background initiator container remains for this `RUN`.

- [ ] **Step 6.2: Stop monitor and target containers on 102**

Run on `lingjun-102`:

```bash
if [ -s "$OUT/raw/rdma-monitor.pid" ]; then
  kill "$(cat "$OUT/raw/rdma-monitor.pid")" 2>/dev/null || true
fi

docker ps --format '{{.Names}}' | grep "^kv_${RUN}_" | xargs -r docker rm -f
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "kv_${RUN}" || true
```

Expected: no target container remains for this `RUN`.

## Task 7: Report Update Through A Subagent

**Files:**
- Read: `docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md`
- Read remotely or from pasted output: `/tmp/kv-transfer-bench/manual/${RUN}/aggregated-summary.csv`
- Read remotely or from pasted output: `/tmp/kv-transfer-bench/manual/${RUN}/raw/rdma-rcv-monitor.csv`
- Future create or modify: `docs/superpowers/reports/2026-06-24-kv-transfer-capped-background-report.md`

- [ ] **Step 7.1: Dispatch report-writing subagent after each completed group**

Use a fresh subagent for report work. Prompt it with this contract:

```text
Write or update a Chinese KV Transfer experiment report.

Style reference:
/home/tiancij/sglang/docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md

Inputs:
- RUN name
- aggregated-summary.csv contents or path
- raw target logs path
- raw initiator logs path
- rdma-rcv-monitor.csv contents or path
- prior report path if updating an existing report

Report requirements:
- Follow the reference report shape: current experiment scope, success-result index, command shape, result files, summary table, observations.
- Include the full 21-row result table from aggregated-summary.csv for every completed RUN.
- The full table must include Size, p50 latency, p90 latency, p99 latency, mean latency, p50 bandwidth, mean bandwidth, and error_count.
- Highlight 512MiB, 1GiB, and 2GiB again in the observation and cross-run comparison sections.
- Mention foreground profile, background level, lane count, shard count, p50/p90/p99 latency, p50 bandwidth, and error_count.
- If charts are added, make the Markdown self-contained so the file can be sent alone.
- Do not claim a completed result unless aggregated-summary.csv exists and error_sum is 0.
- Do not include low-level implementation details unless the user asks for a methods appendix.
```

Expected: the subagent returns the report path and a concise summary of what it added.

## Task 8: Matrix Completion Checklist

- [ ] `200_2x100_bg1_cap100_moonbg` recorded if rerun under Mooncake background.
- [ ] `200_2x100_bg10_cap100_moonbg` recorded if rerun under Mooncake background.
- [ ] `200_2x100_bg50_cap100_moonbg` recorded if rerun under Mooncake background.
- [ ] `200_2x100_bg90_cap100_moonbg` recorded if rerun under Mooncake background.
- [ ] `200_4x50_bg1_cap50_moonbg` run, aggregated, and recorded.
- [ ] `200_4x50_bg10_cap50_moonbg` run, aggregated, and recorded.
- [ ] `200_4x50_bg50_cap50_moonbg` run, aggregated, and recorded.
- [ ] `200_4x50_bg90_cap50_moonbg` run, aggregated, and recorded.
- [ ] `400_2x200_bg1_cap200_moonbg` run, aggregated, and recorded.
- [ ] `400_2x200_bg10_cap200_moonbg` run, aggregated, and recorded.
- [ ] `400_2x200_bg50_cap200_moonbg` run, aggregated, and recorded.
- [ ] `400_2x200_bg90_cap200_moonbg` run, aggregated, and recorded.
- [ ] `400_4x100_bg1_cap100_moonbg` run, aggregated, and recorded.
- [ ] `400_4x100_bg10_cap100_moonbg` run, aggregated, and recorded.
- [ ] `400_4x100_bg50_cap100_moonbg` run, aggregated, and recorded.
- [ ] `400_4x100_bg90_cap100_moonbg` run, aggregated, and recorded.

Final acceptance criteria:

```text
Every run has row_count 21.
Every run has error_sum 0.
Every run has foreground target logs, background target logs, foreground initiator logs, background initiator logs, RDMA counter logs, shard summaries, shard samples, and aggregated summary.
The report has one success-result index entry and one result section per completed RUN.
The report includes a cross-run comparison table for 512MiB, 1GiB, and 2GiB.
```

## Self-Review Notes

- Spec coverage: covers profiles `2x100`, `4x50`, `2x200`, and `4x100`; covers background levels `1/10/50/90`; replaces external background traffic with Mooncake-generated background traffic; covers foreground rate limits; covers report style and subagent handoff.
- Placeholder scan: the only values to change are explicit run variables in Task 1.1; target JSON is copied from live 102 output because it contains runtime pointers and session ids.
- Type consistency: `SHARDS` is used by aggregation; `LANES` controls target and initiator launch; `LANE_CAP` controls foreground `--rate-limit-gbps`; `BG_RATE` controls background `--rate-limit-gbps`.
