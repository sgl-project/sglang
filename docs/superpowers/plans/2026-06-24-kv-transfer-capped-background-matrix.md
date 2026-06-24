# KV Transfer Capped Background Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the next KV Transfer experiment matrix for `2x100`, `4x50`, `2x200`, and `4x100` with foreground measurement rate limits and `1/10/50/90Gbps` per-lane background traffic, then write the results into a report that follows the existing 2026-06-23 report style.

**Architecture:** Treat each experiment as one manual run named `${PROFILE}_bg${BG}_cap${LANE_CAP}` under `/tmp/kv-transfer-bench/manual/`. Node `099` runs the foreground KV initiators, node `102` runs foreground targets and background servers, node `100` runs even-lane background clients, and node `101` runs odd-lane background clients. Each run records raw target logs, background logs, network counters, per-lane foreground samples, and one aggregated logical-transfer summary.

**Tech Stack:** SGLang/Mooncake `kv_transfer_latency.py`, Docker image `sglang-pd-switch:tianciJ`, RDMA IPv6 GID index 3, `ib_write_bw`, host shell, host Python CSV/JSONL aggregation, Markdown reports under `docs/superpowers/reports/`.

---

## Files

- Create: `docs/superpowers/plans/2026-06-24-kv-transfer-capped-background-matrix.md`
- Future create or modify: `docs/superpowers/reports/YYYY-MM-DD-kv-transfer-capped-background-report.md`
- Reference report style: `docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md`
- Remote script used by all Docker runs: `/root/kv_transfer_bench/kv_transfer_latency.py`
- Per-run remote output root: `/tmp/kv-transfer-bench/manual/${RUN}`

## Experiment Matrix

Run all four background levels for each profile.

| Profile | RUN prefix | Lanes | Shards | Foreground cap per lane | Target max bytes per lane | Shard size list |
|---|---|---:|---:|---:|---:|---|
| `2x100` | `200_2x100` | `0 1` | 2 | `100Gbps` | `1GB` | `DENSE_SHARD_SIZES_2` |
| `4x50` | `200_4x50` | `0 1 2 3` | 4 | `50Gbps` | `512MB` | `DENSE_SHARD_SIZES_4` |
| `2x200` | `400_2x200` | `0 1` | 2 | `200Gbps` | `1GB` | `DENSE_SHARD_SIZES_2` |
| `4x100` | `400_4x100` | `0 1 2 3` | 4 | `100Gbps` | `512MB` | `DENSE_SHARD_SIZES_4` |

Background traffic is intentionally set directly to these per-lane values:

```text
BG=1   -> RATE=1Gbps per active lane
BG=10  -> RATE=10Gbps per active lane
BG=50  -> RATE=50Gbps per active lane
BG=90  -> RATE=90Gbps per active lane
```

The foreground cap defines the experiment profile. Do not add the old base-fill formula to the background rate for this matrix.

Run order:

```text
200_2x100_bg1_cap100   complete
200_2x100_bg10_cap100  complete
200_2x100_bg50_cap100  next
200_2x100_bg90_cap100
200_4x50_bg1_cap50
200_4x50_bg10_cap50
200_4x50_bg50_cap50
200_4x50_bg90_cap50
400_2x200_bg1_cap200
400_2x200_bg10_cap200
400_2x200_bg50_cap200
400_2x200_bg90_cap200
400_4x100_bg1_cap100
400_4x100_bg10_cap100
400_4x100_bg50_cap100
400_4x100_bg90_cap100
```

## Common Constants

Use these RDMA addresses throughout.

```bash
# 099 foreground source IPv6 addresses
SRC_IP_0=fd03:4514:80:6240::1
SRC_IP_1=fd03:4514:80:6241::1
SRC_IP_2=fd03:4514:80:6242::1
SRC_IP_3=fd03:4514:80:6243::1

# 102 target IPv6 addresses
TGT_IP_0=fd03:4514:80:5f00::1
TGT_IP_1=fd03:4514:80:5f01::1
TGT_IP_2=fd03:4514:80:5f02::1
TGT_IP_3=fd03:4514:80:5f03::1

# background control-plane target
TGT_IPV4=192.168.0.41
```

Dense size lists:

```bash
export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB
```

Lane placement:

| Lane | 099 source IPv6 | 102 target IPv6 | RDMA device | Background client node |
|---:|---|---|---|---|
| 0 | `fd03:4514:80:6240::1` | `fd03:4514:80:5f00::1` | `mlx5_bond_0` | `100` |
| 1 | `fd03:4514:80:6241::1` | `fd03:4514:80:5f01::1` | `mlx5_bond_1` | `101` |
| 2 | `fd03:4514:80:6242::1` | `fd03:4514:80:5f02::1` | `mlx5_bond_2` | `100` |
| 3 | `fd03:4514:80:6243::1` | `fd03:4514:80:5f03::1` | `mlx5_bond_3` | `101` |

## Task 1: Choose One Run

**Files:**
- Read: this plan
- Create remotely: `/tmp/kv-transfer-bench/manual/${RUN}/raw/`

- [ ] **Step 1.1: Choose the next run**

For the current next run, use:

```bash
PROFILE=200_2x100
BG=50
LANE_CAP=100
RATE=50
LANES=(0 1)
SHARDS=2
MAX_BYTES=1GB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_2"
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
```

For later runs, choose values from this table:

| RUN | `PROFILE` | `BG` | `LANE_CAP` | `RATE` | `LANES` | `SHARDS` | `MAX_BYTES` | `DENSE_SHARD_SIZES` |
|---|---|---:|---:|---:|---|---:|---:|---|
| `200_2x100_bg50_cap100` | `200_2x100` | 50 | 100 | 50 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_2x100_bg90_cap100` | `200_2x100` | 90 | 100 | 90 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `200_4x50_bg1_cap50` | `200_4x50` | 1 | 50 | 1 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg10_cap50` | `200_4x50` | 10 | 50 | 10 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg50_cap50` | `200_4x50` | 50 | 50 | 50 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `200_4x50_bg90_cap50` | `200_4x50` | 90 | 50 | 90 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_2x200_bg1_cap200` | `400_2x200` | 1 | 200 | 1 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg10_cap200` | `400_2x200` | 10 | 200 | 10 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg50_cap200` | `400_2x200` | 50 | 200 | 50 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_2x200_bg90_cap200` | `400_2x200` | 90 | 200 | 90 | `(0 1)` | 2 | `1GB` | `$DENSE_SHARD_SIZES_2` |
| `400_4x100_bg1_cap100` | `400_4x100` | 1 | 100 | 1 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg10_cap100` | `400_4x100` | 10 | 100 | 10 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg50_cap100` | `400_4x100` | 50 | 100 | 50 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |
| `400_4x100_bg90_cap100` | `400_4x100` | 90 | 100 | 90 | `(0 1 2 3)` | 4 | `512MB` | `$DENSE_SHARD_SIZES_4` |

- [ ] **Step 1.2: Create the run directory on every active node**

Run on `099`, `102`, and whichever of `100`/`101` is active:

```bash
cd /root/kv_transfer_bench
mkdir -p "$OUT/raw"
```

Expected:

```text
/tmp/kv-transfer-bench/manual/${RUN}/raw exists on every active node.
```

## Task 2: Start Targets And Background Servers On 102

**Files:**
- Create remotely: `$OUT/raw/target-bond${lane}.log`
- Create remotely: `$OUT/raw/bg-server-bond${lane}.log`
- Create remotely: `$OUT/raw/bg-server-pids.txt`
- Create remotely: `$OUT/raw/netdev-rcv-monitor.csv`

- [ ] **Step 2.1: Configure the run on 102**

Run on `lingjun-102`:

```bash
cd /root/kv_transfer_bench

export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB

PROFILE=200_2x100
BG=50
LANE_CAP=100
RATE=50
LANES=(0 1)
SHARDS=2
MAX_BYTES=1GB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_2"
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT/raw"

declare -A TGT_IPS=(
  [0]=fd03:4514:80:5f00::1
  [1]=fd03:4514:80:5f01::1
  [2]=fd03:4514:80:5f02::1
  [3]=fd03:4514:80:5f03::1
)
```

For later runs, change only `PROFILE/BG/LANE_CAP/RATE/LANES/SHARDS/MAX_BYTES/DENSE_SHARD_SIZES` using Task 1.1.

- [ ] **Step 2.2: Clean only this run if it was already started**

Run on `lingjun-102`:

```bash
docker ps --format '{{.Names}}' | grep "^kv_${RUN}_" | xargs -r docker rm -f

if [ -s "$OUT/raw/bg-server-pids.txt" ]; then
  xargs -r kill < "$OUT/raw/bg-server-pids.txt" 2>/dev/null || true
fi

if [ -s "$OUT/raw/netdev-monitor.pid" ]; then
  kill "$(cat "$OUT/raw/netdev-monitor.pid")" 2>/dev/null || true
fi
```

Expected: no old container or monitor process from the same `RUN` remains.

- [ ] **Step 2.3: Start foreground targets**

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
done

for lane in "${LANES[@]}"; do
  until grep -q 'target_ready=true' "$OUT/raw/target-bond${lane}.log"; do sleep 1; done
done

grep '^TARGET_INFO_JSON=' "$OUT/raw"/target-bond*.log
```

Expected:

```text
One TARGET_INFO_JSON line per active lane.
docker ps shows kv_${RUN}_target_bond${lane} containers Up.
```

- [ ] **Step 2.4: Start background servers**

Run on `lingjun-102`:

```bash
: > "$OUT/raw/bg-server-pids.txt"

for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw \
    -d mlx5_bond_${lane} \
    -x 3 \
    -p "$port" \
    -s 1048576 \
    -q 1 \
    --report_gbits \
    > "$OUT/raw/bg-server-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-server-pids.txt"
done

sleep 2
ss -ltnp | grep -E ':(18500|18501|18502|18503)\b' || true
```

Expected for `2x100` and `2x200`:

```text
LISTEN on 18500 and 18501.
```

Expected for `4x50` and `4x100`:

```text
LISTEN on 18500, 18501, 18502, and 18503.
```

- [ ] **Step 2.5: Start a 102 receive counter monitor**

Run on `lingjun-102`:

```bash
DEVICES=()
for lane in "${LANES[@]}"; do
  DEVICES+=("mlx5_bond_${lane}")
done

echo "ts,dev,rcv_Gbps" > "$OUT/raw/netdev-rcv-monitor.csv"
(
  declare -A last
  for dev in "${DEVICES[@]}"; do
    last[$dev]=$(cat /sys/class/net/$dev/statistics/rx_bytes)
  done
  tlast=$(date +%s)
  while true; do
    sleep 2
    now=$(date +%s)
    ts=$(date -Iseconds)
    dt=$((now - tlast))
    for dev in "${DEVICES[@]}"; do
      cur=$(cat /sys/class/net/$dev/statistics/rx_bytes)
      prev=${last[$dev]}
      gbps=$(awk -v cur="$cur" -v prev="$prev" -v dt="$dt" 'BEGIN{printf "%.3f",(cur-prev)*8/dt/1e9}')
      echo "$ts,$dev,$gbps"
      last[$dev]=$cur
    done
    tlast=$now
  done
) >> "$OUT/raw/netdev-rcv-monitor.csv" &
echo $! > "$OUT/raw/netdev-monitor.pid"
```

Expected:

```bash
sleep 5
tail -n 20 "$OUT/raw/netdev-rcv-monitor.csv"
```

The file should contain one row per active `mlx5_bond_N` every two seconds.

- [ ] **Step 2.6: Print target JSON exports for 099**

Run on `lingjun-102`:

```bash
for lane in "${LANES[@]}"; do
  json=$(sed -n 's/^TARGET_INFO_JSON=//p' "$OUT/raw/target-bond${lane}.log" | tail -1)
  echo "export TARGET_JSON_${lane}='$json'"
done
```

Copy the printed `export TARGET_JSON_${lane}=<runtime JSON>` lines to `lingjun-099`. Do not execute them on `102`; they are for the foreground initiator node.

## Task 3: Start Background Clients On 100 And 101

**Files:**
- Create remotely on `100` or `101`: `$OUT/raw/bg-client-bond${lane}.log`
- Create remotely on `100` or `101`: `$OUT/raw/bg-client-pids.txt`

- [ ] **Step 3.1: Start even lanes on 100**

Run on `lingjun-100` only when the profile includes lane 0 or lane 2:

```bash
cd /root/kv_transfer_bench

PROFILE=200_2x100
BG=50
LANE_CAP=100
RATE=50
LANES=(0)
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
BG_DURATION=1200
mkdir -p "$OUT/raw"

: > "$OUT/raw/bg-client-pids.txt"
for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw 192.168.0.41 \
    -d mlx5_bond_${lane} \
    -x 3 \
    -p "$port" \
    -s 1048576 \
    -q 1 \
    --report_gbits \
    -D "$BG_DURATION" \
    --rate_limit "$RATE" \
    --rate_units g \
    --rate_limit_type SW \
    > "$OUT/raw/bg-client-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-client-pids.txt"
done

sleep 5
ps -ef | grep '[i]b_write_bw' || true
tail -n 20 "$OUT/raw"/bg-client-bond*.log
```

For four-lane profiles on `100`, set:

```bash
LANES=(0 2)
```

Expected: one running `ib_write_bw` process per even lane and no `Couldn't connect` in the logs.

- [ ] **Step 3.2: Start odd lanes on 101**

Run on `lingjun-101` only when the profile includes lane 1 or lane 3:

```bash
cd /root/kv_transfer_bench

PROFILE=200_2x100
BG=50
LANE_CAP=100
RATE=50
LANES=(1)
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
BG_DURATION=1200
mkdir -p "$OUT/raw"

: > "$OUT/raw/bg-client-pids.txt"
for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw 192.168.0.41 \
    -d mlx5_bond_${lane} \
    -x 3 \
    -p "$port" \
    -s 1048576 \
    -q 1 \
    --report_gbits \
    -D "$BG_DURATION" \
    --rate_limit "$RATE" \
    --rate_units g \
    --rate_limit_type SW \
    > "$OUT/raw/bg-client-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-client-pids.txt"
done

sleep 5
ps -ef | grep '[i]b_write_bw' || true
tail -n 20 "$OUT/raw"/bg-client-bond*.log
```

For four-lane profiles on `101`, set:

```bash
LANES=(1 3)
```

Expected: one running `ib_write_bw` process per odd lane and no `Couldn't connect` in the logs.

- [ ] **Step 3.3: Confirm background connections on 102**

Run on `lingjun-102`:

```bash
ss -tnp | grep -E ':(18500|18501|18502|18503)\b' || true
tail -n 20 "$OUT/raw/netdev-rcv-monitor.csv"
```

Expected:

```text
2-lane profiles: ESTAB on 18500 and 18501.
4-lane profiles: ESTAB on 18500, 18501, 18502, and 18503.
102 receive counters are roughly near RATE per active lane before foreground starts.
```

## Task 4: Run Foreground Measurement On 099

**Files:**
- Create remotely: `$OUT/raw/init-bond${lane}-dense.log`
- Create remotely: `$OUT/raw/shard-bond${lane}-dense-summary.csv`
- Create remotely: `$OUT/raw/shard-bond${lane}-dense-samples.jsonl`

- [ ] **Step 4.1: Configure the run on 099**

Run on `lingjun-099`:

```bash
cd /root/kv_transfer_bench

export DENSE_SHARD_SIZES_2=512KB,1MB,2MB,4MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,384MB,512MB,640MB,768MB,896MB,1GB
export DENSE_SHARD_SIZES_4=256KB,512KB,1MB,2MB,4MB,6MB,8MB,12MB,16MB,24MB,32MB,48MB,64MB,96MB,128MB,192MB,256MB,320MB,384MB,448MB,512MB

PROFILE=200_2x100
BG=50
LANE_CAP=100
RATE=50
LANES=(0 1)
SHARDS=2
MAX_BYTES=1GB
DENSE_SHARD_SIZES="$DENSE_SHARD_SIZES_2"
CHUNK_SIZE=64MB
RUN=${PROFILE}_bg${BG}_cap${LANE_CAP}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT/raw"

declare -A SRC_IPS=(
  [0]=fd03:4514:80:6240::1
  [1]=fd03:4514:80:6241::1
  [2]=fd03:4514:80:6242::1
  [3]=fd03:4514:80:6243::1
)
```

Paste the exact `export TARGET_JSON_${lane}=<runtime JSON>` lines printed by `102` in Task 2.6.

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
```

Expected:

```text
One summary_csv line per active lane.
No Traceback, RuntimeError, ret=-1, or failed transfer in the initiator logs.
```

- [ ] **Step 4.3: Check initiator logs**

Run on `lingjun-099`:

```bash
grep -H -E 'Traceback|RuntimeError|failed|ret=-1|Unable|ERROR' "$OUT/raw"/init-bond*-dense.log || true
ls -lh "$OUT/raw"/shard-bond*-dense-summary.csv "$OUT/raw"/shard-bond*-dense-samples.jsonl
```

Expected: no error lines and one summary plus one samples file per active lane.

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
- Read remotely: `$OUT/raw/bg-server-pids.txt`
- Read remotely: `$OUT/raw/netdev-monitor.pid`

- [ ] **Step 6.1: Stop monitor and background servers on 102**

Run on `lingjun-102`:

```bash
if [ -s "$OUT/raw/netdev-monitor.pid" ]; then
  kill "$(cat "$OUT/raw/netdev-monitor.pid")" 2>/dev/null || true
fi

if [ -s "$OUT/raw/bg-server-pids.txt" ]; then
  xargs -r kill < "$OUT/raw/bg-server-pids.txt" 2>/dev/null || true
fi

ss -tnp | grep -E ':(18500|18501|18502|18503)\b' || true
```

Expected: no `ESTAB` rows for this run after background clients exit.

- [ ] **Step 6.2: Stop foreground target containers on 102**

Run on `lingjun-102`:

```bash
docker ps --format '{{.Names}}' | grep "^kv_${RUN}_" | xargs -r docker rm -f
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "kv_${RUN}" || true
```

Expected: no containers for this `RUN`.

- [ ] **Step 6.3: Stop background clients on 100 and 101**

Run on each active background client node:

```bash
if [ -s "$OUT/raw/bg-client-pids.txt" ]; then
  xargs -r kill < "$OUT/raw/bg-client-pids.txt" 2>/dev/null || true
fi
ps -ef | grep '[i]b_write_bw' || true
```

Expected: no `ib_write_bw` client process for this run.

## Task 7: Report Update Through A Subagent

**Files:**
- Read: `docs/superpowers/reports/2026-06-23-kv-transfer-background-traffic-report.md`
- Read remotely or from pasted output: `/tmp/kv-transfer-bench/manual/${RUN}/aggregated-summary.csv`
- Read remotely or from pasted output: `/tmp/kv-transfer-bench/manual/${RUN}/raw/netdev-rcv-monitor.csv`
- Future create or modify: `docs/superpowers/reports/YYYY-MM-DD-kv-transfer-capped-background-report.md`

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
- netdev-rcv-monitor.csv contents or path
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

- [ ] **Step 7.2: Review the report before accepting it**

Check the generated report:

```bash
grep -n "${RUN}" docs/superpowers/reports/*.md
grep -n "512MiB\\|1GiB\\|2GiB" docs/superpowers/reports/*.md | head -40
```

Expected:

```text
The new RUN appears in the success-result index and in a result section.
The completed RUN section contains the full 21-row table, and the three key sizes appear again with exact p50 latency and bw_p50 values.
The report follows the same broad style as the 2026-06-23 reference report.
```

## Task 8: Matrix Completion Checklist

**Files:**
- Read remotely: `/tmp/kv-transfer-bench/manual/*/aggregated-summary.csv`
- Future modify: capped-background report

- [ ] `200_2x100_bg1_cap100` recorded in report.
- [ ] `200_2x100_bg10_cap100` recorded in report.
- [ ] `200_2x100_bg50_cap100` run, aggregated, and recorded.
- [ ] `200_2x100_bg90_cap100` run, aggregated, and recorded.
- [ ] `200_4x50_bg1_cap50` run, aggregated, and recorded.
- [ ] `200_4x50_bg10_cap50` run, aggregated, and recorded.
- [ ] `200_4x50_bg50_cap50` run, aggregated, and recorded.
- [ ] `200_4x50_bg90_cap50` run, aggregated, and recorded.
- [ ] `400_2x200_bg1_cap200` run, aggregated, and recorded.
- [ ] `400_2x200_bg10_cap200` run, aggregated, and recorded.
- [ ] `400_2x200_bg50_cap200` run, aggregated, and recorded.
- [ ] `400_2x200_bg90_cap200` run, aggregated, and recorded.
- [ ] `400_4x100_bg1_cap100` run, aggregated, and recorded.
- [ ] `400_4x100_bg10_cap100` run, aggregated, and recorded.
- [ ] `400_4x100_bg50_cap100` run, aggregated, and recorded.
- [ ] `400_4x100_bg90_cap100` run, aggregated, and recorded.

Final acceptance criteria:

```text
Every run has row_count 21.
Every run has error_sum 0.
Every run has target logs, initiator logs, background logs, netdev counter logs, shard summaries, shard samples, and aggregated summary.
The report has one success-result index entry and one result section per completed RUN.
The report includes a cross-run comparison table for 512MiB, 1GiB, and 2GiB.
```

## Self-Review Notes

- Spec coverage: covers all requested profiles `2x100`, `4x50`, `2x200`, and `4x100`; covers background levels `1/10/50/90`; covers foreground rate limits; covers report style and subagent handoff.
- Placeholder scan: the only values to change are explicit run variables in Task 1.1; target JSON is copied from live 102 output because it contains runtime pointers and session ids.
- Type consistency: `SHARDS` is used by aggregation; `LANES` controls target/server/initiator launch; `LANE_CAP` controls foreground `--rate-limit-gbps`; `RATE` controls background `ib_write_bw`.
