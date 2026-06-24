# RDMA KV 手动实验 Runbook

目标：手动测量 099 -> 102 的 KV Cache RDMA 传输，在 800G/400G/200G 档位下叠加背景流量，并保存可拟合的 CSV/JSONL 结果。

当前执行入口：

1. 先做第 5 节 `bond2/bond3 smoke`，确认四条 RDMA bond 都能跑 KV transfer。
2. 再按第 7 节顺序跑正式实验。
3. 每跑完一个实验，把 `aggregated-summary.csv` 和 `fit-summary.csv` 发回来，我把结果写进中文报告。

## 0. 实验数量

按当前需求完整展开是 24 个实验，不是 20 个：

| profile | 含义 | lane | 每 lane 目标容量 | 背景占用 |
|---|---:|---|---:|---|
| `800_4x200` | 800G | bond0,bond1,bond2,bond3 | 200G | 1/10/50/90% |
| `400_2x200` | 400G | bond0,bond1 | 200G | 1/10/50/90% |
| `400_4x100` | 400G | bond0,bond1,bond2,bond3 | 100G | 1/10/50/90% |
| `200_1x200` | 200G | bond0 | 200G | 1/10/50/90% |
| `200_2x100` | 200G | bond0,bond1 | 100G | 1/10/50/90% |
| `200_4x50` | 200G | bond0,bond1,bond2,bond3 | 50G | 1/10/50/90% |

如果只先做 20 个，先跳过 `200_4x50` 的 4 个实验。

## 1. 机器分工

| 角色 | 机器 | 用途 |
|---|---|---|
| KV initiator | `lingjun-099` | 前景 KV source |
| KV target | `lingjun-102` | 前景 KV target + 背景流量 target |
| background source A | `lingjun-100` | 背景流量 source，跑 bond0/bond2 |
| background source B | `lingjun-101` | 背景流量 source，跑 bond1/bond3 |

前景 KV transfer 使用 Docker 镜像 `sglang-pd-switch:tianciJ`，只需要 099 和 102 有这个镜像。背景流量用宿主机 `ib_write_bw`，100/101 不需要这个 Docker 镜像。

注意：前景 KV 的 Mooncake 连接使用 RDMA IPv6 `fd03:...`；背景流量 `ib_write_bw` 的控制面连接使用 102 的 eth0 IPv4 `192.168.0.41`，RDMA 数据面仍由 `-d mlx5_bond_N -x 3` 选择对应 bond。

## 2. RDMA IPv6 地址

099:

```bash
SRC_IPS[0]=fd03:4514:80:6240::1
SRC_IPS[1]=fd03:4514:80:6241::1
SRC_IPS[2]=fd03:4514:80:6242::1
SRC_IPS[3]=fd03:4514:80:6243::1
```

102:

```bash
TGT_IPS[0]=fd03:4514:80:5f00::1
TGT_IPS[1]=fd03:4514:80:5f01::1
TGT_IPS[2]=fd03:4514:80:5f02::1
TGT_IPS[3]=fd03:4514:80:5f03::1
```

所有 bond RDMA 实验都固定：

```bash
MC_USE_IPV6=1
MC_GID_INDEX=3
```

## 3. 背景流量速率表

每条物理 lane 按 200G 算。`4*100` / `2*100` / `4*50` 用背景流量先占掉多余带宽，再在剩余目标带宽里叠加 1/10/50/90% 背景。

| 每 lane 目标容量 | bg1 | bg10 | bg50 | bg90 |
|---:|---:|---:|---:|---:|
| 200G | 2G | 20G | 100G | 180G |
| 100G | 101G | 110G | 150G | 190G |
| 50G | 150.5G | 155G | 175G | 195G |

公式：

```text
rate_per_lane = (200 - lane_cap) + lane_cap * bg_percent
```

如果 `ib_write_bw` 不接受 `150.5` 这种小数，把 `150.5G` 改成 `151G`。

## 4. 每个实验的变量

每次只跑一个 `RUN`。四个终端都设置同一组 `PROFILE/BG/RATE/LANES`。

### 800_4x200

```bash
PROFILE=800_4x200
LANES=(0 1 2 3)
LANE_CAP=200
SHARD_SIZES=256KB:512MB:x2
MAX_BYTES=512MB
```

### 400_2x200

```bash
PROFILE=400_2x200
LANES=(0 1)
LANE_CAP=200
SHARD_SIZES=512KB:1GB:x2
MAX_BYTES=1GB
```

### 400_4x100

```bash
PROFILE=400_4x100
LANES=(0 1 2 3)
LANE_CAP=100
SHARD_SIZES=256KB:512MB:x2
MAX_BYTES=512MB
```

### 200_1x200

```bash
PROFILE=200_1x200
LANES=(0)
LANE_CAP=200
SHARD_SIZES=1MB:2GB:x2
MAX_BYTES=2GB
```

### 200_2x100

```bash
PROFILE=200_2x100
LANES=(0 1)
LANE_CAP=100
SHARD_SIZES=512KB:1GB:x2
MAX_BYTES=1GB
```

### 200_4x50

```bash
PROFILE=200_4x50
LANES=(0 1 2 3)
LANE_CAP=50
SHARD_SIZES=256KB:512MB:x2
MAX_BYTES=512MB
```

背景变量四选一：

```bash
BG=1    # RATE 按上表填
BG=10
BG=50
BG=90
```

然后设置：

```bash
RATE=20
RUN=${PROFILE}_bg${BG}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
BG_DURATION=1200
mkdir -p "$OUT"/raw
```

### 4.1 100/101 背景流量 lane 分配

正式实验时，102 启动所有本轮 lane 的 background server；100 和 101 只启动自己负责的 background client：

| profile | 100 上的 `LANES=(...)` | 101 上的 `LANES=(...)` |
|---|---|---|
| `800_4x200` | `LANES=(0 2)` | `LANES=(1 3)` |
| `400_2x200` | `LANES=(0)` | `LANES=(1)` |
| `400_4x100` | `LANES=(0 2)` | `LANES=(1 3)` |
| `200_1x200` | `LANES=(0)` | 不启动背景 client |
| `200_2x100` | `LANES=(0)` | `LANES=(1)` |
| `200_4x50` | `LANES=(0 2)` | `LANES=(1 3)` |

### 4.2 RATE 快速对照

| profile | bg1 | bg10 | bg50 | bg90 |
|---|---:|---:|---:|---:|
| `800_4x200` | `RATE=2` | `RATE=20` | `RATE=100` | `RATE=180` |
| `400_2x200` | `RATE=2` | `RATE=20` | `RATE=100` | `RATE=180` |
| `400_4x100` | `RATE=101` | `RATE=110` | `RATE=150` | `RATE=190` |
| `200_1x200` | `RATE=2` | `RATE=20` | `RATE=100` | `RATE=180` |
| `200_2x100` | `RATE=101` | `RATE=110` | `RATE=150` | `RATE=190` |
| `200_4x50` | `RATE=150.5` | `RATE=155` | `RATE=175` | `RATE=195` |

## 5. 第一次必须先 smoke bond2/bond3

bond0 和 bond1 已经跑通过。跑完整矩阵前，先验证 bond2/bond3。

在 102 启动 bond2 target：

```bash
cd /root/kv_transfer_bench
OUT=/tmp/kv-transfer-bench/manual/smoke_bond2/raw
mkdir -p "$OUT"
docker run --rm --name kv_smoke_bond2_target \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
  -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_2 -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
  -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
  -v "$OUT":/tmp/kv-transfer-bench \
  -v /dev/infiniband:/dev/infiniband \
  sglang-pd-switch:tianciJ \
  bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role target --host fd03:4514:80:5f02::1 --gpu-id 2 --ib-device mlx5_bond_2 --protocol rdma --max-bytes 256MB --target-info-file /tmp/kv-transfer-bench/target.json"
```

把 102 输出里的 `TARGET_INFO_JSON=...` 复制到 099，然后在 099 跑：

```bash
cd /root/kv_transfer_bench
export TARGET_INFO_JSON='粘贴102输出的JSON'
OUT=/tmp/kv-transfer-bench/manual/smoke_bond2/raw
mkdir -p "$OUT"
docker run --rm --name kv_smoke_bond2_init \
  --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
  -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_2 -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
  -e TARGET_INFO_JSON="$TARGET_INFO_JSON" \
  -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
  -v "$OUT":/tmp/kv-transfer-bench \
  -v /dev/infiniband:/dev/infiniband \
  sglang-pd-switch:tianciJ \
  bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role initiator --host fd03:4514:80:6242::1 --gpu-id 2 --ib-device mlx5_bond_2 --protocol rdma --sizes 1MB --warmup 1 --repeat 3 --summary-csv /tmp/kv-transfer-bench/summary.csv --samples-jsonl /tmp/kv-transfer-bench/samples.jsonl"
```

bond3 同理，把 `mlx5_bond_2` 改成 `mlx5_bond_3`，IP 改成：

```text
099: fd03:4514:80:6243::1
102: fd03:4514:80:5f03::1
GPU: 3
```

## 6. 正式实验步骤

下面以一个 profile/bg 为单位跑。四台机器的变量必须一致。

### 6.1 在 102 启动 KV targets

```bash
cd /root/kv_transfer_bench
declare -a TGT_IPS
TGT_IPS[0]=192.168.0.41
TGT_IPS[1]=192.168.0.41
TGT_IPS[2]=192.168.0.41
TGT_IPS[3]=192.168.0.41

# 先粘贴本轮 profile 变量，例如：
# PROFILE=800_4x200
# LANES=(0 1 2 3)
# SHARD_SIZES=256KB:512MB:x2
# MAX_BYTES=512MB
# BG=10
# RATE=20
RUN=${PROFILE}_bg${BG}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT"/raw

for lane in "${LANES[@]}"; do
  name=kv_${RUN}_target_bond${lane}
  docker rm -f "$name" >/dev/null 2>&1 || true
  nohup docker run --rm --name "$name" \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_${lane} -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role target --host ${TGT_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --max-bytes ${MAX_BYTES} --target-info-file /tmp/kv-transfer-bench/target-bond${lane}.json" \
    > "$OUT/raw/target-bond${lane}.log" 2>&1 &
done

for lane in "${LANES[@]}"; do
  until grep -q 'target_ready=true' "$OUT/raw/target-bond${lane}.log"; do
    sleep 1
  done
  grep '^TARGET_INFO_JSON=' "$OUT/raw/target-bond${lane}.log"
done
```

把每个 `TARGET_INFO_JSON=...` 复制到 099。比如：

```bash
export TARGET_JSON_0='{"bytes":...,"session_id":"[fd03:4514:80:5f00::1]:..."}'
export TARGET_JSON_1='{"bytes":...,"session_id":"[fd03:4514:80:5f01::1]:..."}'
```

### 6.2 在 102 启动背景流量 servers

```bash
mkdir -p "$OUT/raw"
: > "$OUT/raw/bg-server-pids.txt"

for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw \
    -d mlx5_bond_${lane} -x 3 --ipv6 -p "$port" \
    -s 1048576 -q 1 --report_gbits \
    > "$OUT/raw/bg-server-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-server-pids.txt"
done
```

### 6.3 在 100 启动背景 clients for bond0/bond2

只保留本轮 `LANES` 里存在的 lane。例如 `LANES=(0 1)` 时，100 只跑 `LANES=(0)`。

```bash
declare -a TGT_IPS
TGT_IPS[0]=fd03:4514:80:5f00::1
TGT_IPS[1]=fd03:4514:80:5f01::1
TGT_IPS[2]=fd03:4514:80:5f02::1
TGT_IPS[3]=fd03:4514:80:5f03::1

# 按本轮选择，例如 800_4x200 跑 0 和 2：
LANES=(0 2)
PROFILE=800_4x200
BG=10
RATE=20
RUN=${PROFILE}_bg${BG}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
BG_DURATION=1200
mkdir -p "$OUT/raw"
: > "$OUT/raw/bg-client-pids.txt"

for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw "${TGT_IPS[$lane]}" \
    -d mlx5_bond_${lane} -x 3 -p "$port" \
    -s 1048576 -q 1 --report_gbits -D "$BG_DURATION" \
    --rate_limit "$RATE" --rate_units g --rate_limit_type SW \
    > "$OUT/raw/bg-client-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-client-pids.txt"
done
```

### 6.4 在 101 启动背景 clients for bond1/bond3

和 100 一样，只是本轮 lane 换成 `1/3`。

```bash
# 例如 800_4x200 跑 1 和 3；400_2x200 只跑 1。
LANES=(1 3)
PROFILE=800_4x200
BG=10
RATE=20
RUN=${PROFILE}_bg${BG}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
BG_DURATION=1200
mkdir -p "$OUT/raw"
: > "$OUT/raw/bg-client-pids.txt"

declare -a TGT_IPS
TGT_IPS[0]=192.168.0.41
TGT_IPS[1]=192.168.0.41
TGT_IPS[2]=192.168.0.41
TGT_IPS[3]=192.168.0.41

for lane in "${LANES[@]}"; do
  port=$((18500 + lane))
  nohup ib_write_bw "${TGT_IPS[$lane]}" \
    -d mlx5_bond_${lane} -x 3 -p "$port" \
    -s 1048576 -q 1 --report_gbits -D "$BG_DURATION" \
    --rate_limit "$RATE" --rate_units g --rate_limit_type SW \
    > "$OUT/raw/bg-client-bond${lane}.log" 2>&1 &
  echo $! >> "$OUT/raw/bg-client-pids.txt"
done
```

### 6.5 在 099 启动 KV initiators

```bash
cd /root/kv_transfer_bench
declare -a SRC_IPS
SRC_IPS[0]=fd03:4514:80:6240::1
SRC_IPS[1]=fd03:4514:80:6241::1
SRC_IPS[2]=fd03:4514:80:6242::1
SRC_IPS[3]=fd03:4514:80:6243::1

# 粘贴本轮 profile 变量
# PROFILE=800_4x200
# LANES=(0 1 2 3)
# SHARD_SIZES=256KB:512MB:x2
# BG=10
# RATE=20
RUN=${PROFILE}_bg${BG}
OUT=/tmp/kv-transfer-bench/manual/${RUN}
mkdir -p "$OUT/raw"

# 粘贴 102 打印出的 JSON
# export TARGET_JSON_0='...'
# export TARGET_JSON_1='...'
# export TARGET_JSON_2='...'
# export TARGET_JSON_3='...'

for lane in "${LANES[@]}"; do
  json_var=TARGET_JSON_${lane}
  target_json=${!json_var}
  name=kv_${RUN}_init_bond${lane}
  docker rm -f "$name" >/dev/null 2>&1 || true
  nohup docker run --rm --name "$name" \
    --gpus all --network host --ipc host --privileged --ulimit memlock=-1:-1 \
    -e MOONCAKE_PROTOCOL=rdma -e IB_DEVICE=mlx5_bond_${lane} -e MC_USE_IPV6=1 -e MC_GID_INDEX=3 \
    -e TARGET_INFO_JSON="$target_json" \
    -v /root/kv_transfer_bench:/workspace/kv_transfer_bench:ro \
    -v "$OUT/raw":/tmp/kv-transfer-bench \
    -v /dev/infiniband:/dev/infiniband \
    sglang-pd-switch:tianciJ \
    bash -lc "cd /workspace/kv_transfer_bench && python3 kv_transfer_latency.py --role initiator --host ${SRC_IPS[$lane]} --gpu-id ${lane} --ib-device mlx5_bond_${lane} --protocol rdma --sizes ${SHARD_SIZES} --warmup 3 --repeat 20 --summary-csv /tmp/kv-transfer-bench/shard-bond${lane}-summary.csv --samples-jsonl /tmp/kv-transfer-bench/shard-bond${lane}-samples.jsonl" \
    > "$OUT/raw/init-bond${lane}.log" 2>&1 &
done

wait
grep -H 'summary_csv=' "$OUT"/raw/init-bond*.log
```

### 6.6 在 099 聚合多 shard 结果并拟合

聚合逻辑：

```text
同一次 logical KV request 被切成 N 个 shard。
每个 shard 走一条 bond。
logical latency = 同一 iteration 下 N 个 shard latency 的最大值。
logical bytes = shard bytes * shard 数。
```

拟合逻辑：

```text
画完整曲线使用 1MB -> 2GB 全部点。
拟合 latency_ms = fixed_overhead_ms + bytes / bandwidth 只使用 logical bytes >= 64MB。
如果后续看到 64MB 点抖动明显，再把阈值改成 128MB。
```

```bash
python3 - "$OUT" "${#LANES[@]}" <<'PY'
import csv, glob, json, math, statistics, sys
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

groups = defaultdict(list)
errors = defaultdict(int)
for path in sorted(glob.glob(str(raw / "shard-bond*-samples.jsonl"))):
    for line in open(path):
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
    p50 = pct(vals, 0.50)
    mean = statistics.fmean(vals)
    summary.append({
        "bytes": logical_bytes,
        "human_bytes": f"{logical_bytes / 1024 / 1024:.2f}MiB",
        "repeat_count": len(vals),
        "error_count": errors[logical_bytes],
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p90": pct(vals, 0.90),
        "latency_ms_p99": pct(vals, 0.99),
        "latency_ms_min": min(vals),
        "latency_ms_max": max(vals),
        "bandwidth_GiBps_p50": (logical_bytes / 1024**3) / (p50 / 1000),
        "bandwidth_GiBps_mean": (logical_bytes / 1024**3) / (mean / 1000),
    })

with open(out / "aggregated-summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

fit_rows = [r for r in summary if int(r["bytes"]) >= 64 * 1024**2]
n = len(fit_rows)
x = [float(r["bytes"]) for r in fit_rows]
y = [float(r["latency_ms_p50"]) for r in fit_rows]
sx, sy = sum(x), sum(y)
sxx = sum(v * v for v in x)
sxy = sum(a * b for a, b in zip(x, y))
slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
intercept = (sy - slope * sx) / n
bw_gibps = 1000 / (slope * 1024**3)

with open(out / "fit-summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fixed_overhead_ms", "bandwidth_GiBps", "fit_min_bytes", "fit_points"])
    writer.writeheader()
    writer.writerow({
        "fixed_overhead_ms": intercept,
        "bandwidth_GiBps": bw_gibps,
        "fit_min_bytes": 64 * 1024**2,
        "fit_points": n,
    })

print(out / "aggregated-summary.csv")
print(out / "fit-summary.csv")
PY
```

### 6.7 清理本轮进程

099:

```bash
docker ps --format '{{.Names}}' | grep "^kv_${RUN}_init_" | xargs -r docker rm -f
```

102:

```bash
docker ps --format '{{.Names}}' | grep "^kv_${RUN}_target_" | xargs -r docker rm -f
xargs -r kill < "$OUT/raw/bg-server-pids.txt" 2>/dev/null || true
```

100/101:

```bash
xargs -r kill < "$OUT/raw/bg-client-pids.txt" 2>/dev/null || true
```

## 7. 推荐执行顺序

先跑无争议、最容易验证的 20 个：

```text
1.  200_1x200 bg1
2.  200_1x200 bg10
3.  200_1x200 bg50
4.  200_1x200 bg90
5.  400_2x200 bg1
6.  400_2x200 bg10
7.  400_2x200 bg50
8.  400_2x200 bg90
9.  800_4x200 bg1
10. 800_4x200 bg10
11. 800_4x200 bg50
12. 800_4x200 bg90
13. 200_2x100 bg1
14. 200_2x100 bg10
15. 200_2x100 bg50
16. 200_2x100 bg90
17. 400_4x100 bg1
18. 400_4x100 bg10
19. 400_4x100 bg50
20. 400_4x100 bg90
```

最后补 4 个压力最大的：

```text
21. 200_4x50 bg1
22. 200_4x50 bg10
23. 200_4x50 bg50
24. 200_4x50 bg90
```

每跑完一个实验，把这两个文件内容发回来：

```bash
cat "$OUT/aggregated-summary.csv"
cat "$OUT/fit-summary.csv"
```

如果失败，把这几个日志发回来：

```bash
ls -lh "$OUT/raw"
tail -n 80 "$OUT/raw"/target-bond*.log
tail -n 80 "$OUT/raw"/init-bond*.log
tail -n 80 "$OUT/raw"/bg-*.log
```
