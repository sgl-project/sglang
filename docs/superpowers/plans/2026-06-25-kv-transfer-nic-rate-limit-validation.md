# KV Transfer NIC Rate Limit Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate whether hardware NIC/link speed limiting can replace capfill traffic for Mooncake RDMA background-load experiments.

**Architecture:** First run a single-port manual validation on `lingjun-102` by lowering `mlx5_bond_0` from 200G to 100G with `mlxlink`, then drive one long Mooncake RDMA flow from `lingjun-099` and measure receiver-side `port_rcv_data`. Only if this proves the cap is real should the auto experiment script be extended to run bg-only capped-load matrices.

**Tech Stack:** NVIDIA MFT `mst`/`mlxlink`, Mooncake RDMA benchmark scripts in `/root/kv_transfer_bench`, Docker image `sglang-pd-switch:tianciJ`, RDMA counters under `/sys/class/infiniband`.

---

## Scope

This plan intentionally does not modify the auto experiment script yet. It validates one target-side NIC first:

- Target/receiver: `lingjun-102`, SSH target `root@192.168.0.41`.
- Initiator/sender: `lingjun-099`.
- Device under test: `mlx5_bond_0`.
- MST device under test: `/dev/mst/mt41692_pciconf0`.
- Netdev mapping: `mlx5_bond_0 port 1 ==> bond0`.
- Test cap: 100G.
- Test traffic: one long Mooncake RDMA flow for 60 seconds.
- Success condition: receiver-side `mlx5_bond_0` `rcv_Gbps` stays around 95-105Gbps during the long flow.

## Files

- Create: `docs/superpowers/plans/2026-06-25-kv-transfer-nic-rate-limit-validation.md`
- Read on 099: `/root/kv_transfer_bench/kv_transfer_latency.py`
- Runtime output on 102: `/tmp/kv-transfer-bench/manual/manual_niccap_100g_bond0_*/raw/*`
- Runtime output on 099: `/tmp/kv-transfer-bench/manual/manual_niccap_100g_bond0/raw/*`

## Task 1: Save Current Link State

- [ ] **Step 1: Save 102 link state before changing anything**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
mst start
mkdir -p /tmp/kv-transfer-bench/nic-link-state-before
for DEV in /dev/mst/mt41692_pciconf0 /dev/mst/mt41692_pciconf1 /dev/mst/mt41692_pciconf2 /dev/mst/mt41692_pciconf3; do
  base=$(basename "$DEV")
  mlxlink -d "$DEV" > "/tmp/kv-transfer-bench/nic-link-state-before/${base}.txt"
done
mst status -v > /tmp/kv-transfer-bench/nic-link-state-before/mst-status-v.txt
ibdev2netdev > /tmp/kv-transfer-bench/nic-link-state-before/ibdev2netdev.txt
ls -lh /tmp/kv-transfer-bench/nic-link-state-before/
'
```

Expected: files are created under `/tmp/kv-transfer-bench/nic-link-state-before/`.

- [ ] **Step 2: Print the current active speed for `mlx5_bond_0`**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
DEV=/dev/mst/mt41692_pciconf0
mlxlink -d "$DEV" | grep -E "State|Speed|Width|Physical|Supported|Enabled|Active"
'
```

Expected: the output shows the current link is up and active near 200G before the test.

## Task 2: Apply a 100G Link Cap on 102 `mlx5_bond_0`

- [ ] **Step 1: Try `100G_2X` first**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
DEV=/dev/mst/mt41692_pciconf0

echo "=== set 100G_2X ==="
mlxlink -d "$DEV" --speeds 100G_2X --link_mode_force

sleep 5

echo "=== after 100G_2X ==="
mlxlink -d "$DEV" | grep -E "State|Speed|Width|Physical|Supported|Enabled|Active"
'
```

Expected: active speed is 100G. If this command fails or the link remains 200G, do Step 2.

- [ ] **Step 2: If needed, try `100G_4X`**

Run on `lingjun-099` only if Step 1 did not produce a 100G active speed:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
DEV=/dev/mst/mt41692_pciconf0

echo "=== set 100G_4X ==="
mlxlink -d "$DEV" --speeds 100G_4X --link_mode_force

sleep 5

echo "=== after 100G_4X ==="
mlxlink -d "$DEV" | grep -E "State|Speed|Width|Physical|Supported|Enabled|Active"
'
```

Expected: active speed is 100G. If `mlxlink` reports `Supported Speeds Are: 200G_2X`, stop this link-speed approach because the cable/module does not support downshifting this link to 100G.

## Task 3: Start the 102 Target and Receiver-Side Monitor

- [ ] **Step 1: Start one target on 102**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
cd /root/kv_transfer_bench

RUN=manual_niccap_100g_bond0_$(date +%Y%m%d_%H%M%S)
OUT=/tmp/kv-transfer-bench/manual/$RUN
mkdir -p "$OUT/raw"
echo "$RUN" > /tmp/kv-transfer-bench/manual/latest-niccap-run.txt

docker ps -a --format "{{.Names}}" | grep "^kv_${RUN}_" | xargs -r docker rm -f

nohup docker run --rm --name "kv_${RUN}_target" \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
  -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_0 -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
  -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
  -v "$OUT/raw":/tmp/kv-transfer-bench \
  -v /dev/infiniband:/dev/infiniband \
  sglang-pd-switch:tianciJ bash -lc \
  "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role target --host fd03:4514:80:5f00::1 --gpu-id 0 --ib-device mlx5_bond_0 --protocol rdma --max-bytes 2GB --target-info-file /tmp/kv-transfer-bench/target.json" \
  > "$OUT/raw/target.log" 2>&1 &

until grep -q "target_ready=true" "$OUT/raw/target.log"; do sleep 1; done
grep "^TARGET_INFO_JSON=" "$OUT/raw/target.log"
'
```

Expected: output contains one `TARGET_INFO_JSON=...` line.

- [ ] **Step 2: Start receiver-side RDMA monitor on 102**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
RUN=$(cat /tmp/kv-transfer-bench/manual/latest-niccap-run.txt)
OUT=/tmp/kv-transfer-bench/manual/$RUN

(
  echo "ts,dev,rcv_Gbps"
  dev=mlx5_bond_0
  last=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
  tlast=$(date +%s%N)
  while true; do
    sleep 2
    cur=$(cat /sys/class/infiniband/$dev/ports/1/counters/port_rcv_data)
    now=$(date +%s%N)
    awk -v ts="$(date -Iseconds)" -v dev="$dev" -v cur="$cur" -v last="$last" -v now="$now" -v tlast="$tlast" \
      "BEGIN { printf \"%s,%s,%.3f\\n\", ts, dev, ((cur-last)*32)/(now-tlast) }"
    last=$cur
    tlast=$now
  done
) >> "$OUT/raw/rdma-rcv-monitor.csv" &

echo $! > "$OUT/raw/rdma-monitor.pid"
tail -n 5 "$OUT/raw/rdma-rcv-monitor.csv"
'
```

Expected: `rdma-rcv-monitor.csv` exists on 102 and starts receiving samples.

## Task 4: Run a 60-Second Mooncake RDMA Flow from 099

- [ ] **Step 1: Fetch target JSON from 102 and run one long flow**

Run on `lingjun-099`:

```bash
cd /root/kv_transfer_bench

RUN=manual_niccap_100g_bond0
OUT=/tmp/kv-transfer-bench/manual/$RUN
mkdir -p "$OUT/raw"

TARGET_JSON=$(ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
RUN=$(cat /tmp/kv-transfer-bench/manual/latest-niccap-run.txt)
OUT=/tmp/kv-transfer-bench/manual/$RUN
sed -n "s/^TARGET_INFO_JSON=//p" "$OUT/raw/target.log" | tail -1
')

docker run --rm --name "kv_${RUN}_init" \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
  -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_0 -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
  -e TARGET_INFO_JSON="$TARGET_JSON" \
  -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
  -v "$OUT/raw":/tmp/kv-transfer-bench \
  -v /dev/infiniband:/dev/infiniband \
  sglang-pd-switch:tianciJ bash -lc \
  "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role initiator --host fd03:4514:80:5f00::2 --gpu-id 0 --ib-device mlx5_bond_0 --protocol rdma --sizes 2GB --background-bytes 2GB --background-duration-seconds 60 --chunk-size 16MB --flow-id niccap-test --summary-csv /tmp/kv-transfer-bench/niccap-summary.csv --samples-jsonl /tmp/kv-transfer-bench/niccap-samples.jsonl"
```

Expected: the command runs for about 60 seconds and writes `/tmp/kv-transfer-bench/manual/manual_niccap_100g_bond0/raw/niccap-summary.csv` on 099.

## Task 5: Analyze Receiver-Side Bandwidth

- [ ] **Step 1: Copy receiver monitor CSV back to 099**

Run on `lingjun-099`:

```bash
RUN_LOCAL=manual_niccap_100g_bond0
OUT_LOCAL=/tmp/kv-transfer-bench/manual/$RUN_LOCAL
mkdir -p "$OUT_LOCAL/raw"

REMOTE_RUN=$(ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 'cat /tmp/kv-transfer-bench/manual/latest-niccap-run.txt')

scp -i ~/.ssh/kvbench_102 \
  "root@192.168.0.41:/tmp/kv-transfer-bench/manual/$REMOTE_RUN/raw/rdma-rcv-monitor.csv" \
  "$OUT_LOCAL/raw/rdma-rcv-monitor.csv"
```

Expected: monitor CSV is present at `/tmp/kv-transfer-bench/manual/manual_niccap_100g_bond0/raw/rdma-rcv-monitor.csv`.

- [ ] **Step 2: Summarize samples above 1Gbps**

Run on `lingjun-099`:

```bash
python3 - <<'PY'
import csv
import statistics

p = "/tmp/kv-transfer-bench/manual/manual_niccap_100g_bond0/raw/rdma-rcv-monitor.csv"
vals = []
with open(p) as f:
    for r in csv.DictReader(f):
        v = float(r["rcv_Gbps"])
        if v > 1:
            vals.append(v)

print("sample_count", len(vals))
if vals:
    print("mean_Gbps", round(statistics.mean(vals), 2))
    print("p50_Gbps", round(statistics.median(vals), 2))
    print("max_Gbps", round(max(vals), 2))
PY
```

Expected for successful hardware/link cap: `p50_Gbps` is near 100. If it remains near 180-210, the attempted NIC/link cap did not affect the RDMA path.

## Task 6: Restore Link Speed and Clean Up

- [ ] **Step 1: Restore 102 `mlx5_bond_0` to 200G**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set -euo pipefail
DEV=/dev/mst/mt41692_pciconf0

echo "=== restore 200G_2X ==="
mlxlink -d "$DEV" --speeds 200G_2X || true

sleep 5

echo "=== after restore ==="
mlxlink -d "$DEV" | grep -E "State|Speed|Width|Physical|Supported|Enabled|Active"
'
```

Expected: active speed is back near `200G` with `Width : 2x`.

- [ ] **Step 2: Stop monitor and target containers**

Run on `lingjun-099`:

```bash
ssh -i ~/.ssh/kvbench_102 root@192.168.0.41 '
set +e
RUN=$(cat /tmp/kv-transfer-bench/manual/latest-niccap-run.txt 2>/dev/null)
OUT=/tmp/kv-transfer-bench/manual/$RUN

if [ -s "$OUT/raw/rdma-monitor.pid" ]; then
  kill "$(cat "$OUT/raw/rdma-monitor.pid")" 2>/dev/null || true
fi

docker ps -a --format "{{.Names}}" | grep "^kv_${RUN}_" | xargs -r docker rm -f
'
```

Expected: no `kv_manual_niccap_100g_bond0_*` containers remain on 102.

## Task 7: Decision Gate

- [ ] **Step 1: If the 100G cap works, define the next auto-run design**

Use the result from Task 5:

```text
If p50_Gbps is about 95-105:
  Treat hardware/link cap as usable for the next experiment.

If p50_Gbps is about 180-210:
  Do not use this mechanism for RDMA experiments.
  Continue with switch-side QoS/rate limit investigation instead.
```

- [ ] **Step 2: If usable, extend the next experiment matrix**

The next script change should implement:

```text
2x100:
  Set selected ports to 100G, run bg 1/10/50/90, foreground uncapped.

4x50:
  Only implement if 50G link mode is supported and verified.

4x100:
  Set selected ports to 100G, run bg 1/10/50/90, foreground uncapped.

2x200:
  Leave selected ports at 200G, run bg 1/10/50/90, foreground uncapped.
```

Each auto-run must save:

```text
aggregated-summary.csv
raw/*samples.jsonl
raw/*summary.csv
raw/rdma-rcv-monitor.csv copied back from 102
link state before and after the run
```
