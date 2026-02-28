#!/bin/bash
# B03_compare: 对比 B01 / B02 / B04 效率
# 用法: ./B03_compare.sh [--all] [--b01 FILE] [--b02 FILE] [--b04 FILE] [--resolution WxH]
#   默认: B01 vs B02
#   --all: 包含 B04 (4 GPU)，输出 1/2/4 卡对比表
#   --b01/--b02/--b04: 指定结果文件

set -e

BENCH_BASE="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark"
B01_DIR="${BENCH_BASE}/B01_single_gpu"
B02_DIR="${BENCH_BASE}/B02_multiple_gpu"
B04_DIR="${BENCH_BASE}/B04_multiple_gpu_4"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_FILE="${BENCH_BASE}/comparison_B01_B02_B04_${TIMESTAMP}.md"

B01_FILE=""
B02_FILE=""
B04_FILE=""
RESOLUTION_FILTER=""
INCLUDE_B04=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) INCLUDE_B04=true; shift ;;
        --b01) B01_FILE="$2"; shift 2 ;;
        --b02) B02_FILE="$2"; shift 2 ;;
        --b04) B04_FILE="$2"; INCLUDE_B04=true; shift 2 ;;
        --resolution) RESOLUTION_FILTER="$2"; shift 2 ;;
        -h|--help)
            echo "用法: ./B03_compare.sh [--all] [--b01 FILE] [--b02 FILE] [--b04 FILE] [--resolution WxH]"
            echo "  默认: B01 vs B02"
            echo "  --all: 包含 B04 (4 GPU)"
            echo "  --b01/--b02/--b04: 指定结果文件"
            echo "  --resolution: 按分辨率匹配"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

cd /data/users/yandache/workspaces/sglang
[ -f "env_sglang/bin/activate" ] && source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

python3 - "$B01_DIR" "$B02_DIR" "$B04_DIR" "$OUT_FILE" "$B01_FILE" "$B02_FILE" "$B04_FILE" "$RESOLUTION_FILTER" "$INCLUDE_B04" << 'EOFPY'
import json
import os
import sys
from pathlib import Path

b01_dir, b02_dir, b04_dir, out_path, b01_file, b02_file, b04_file, res_filter, include_b04 = sys.argv[1:10]
include_b04 = include_b04.lower() in ("true", "1", "yes")

def pick_file(dir_path, specified, res_filter):
    if specified and os.path.isfile(specified):
        return specified
    dir_p = Path(dir_path)
    files = sorted(dir_p.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    if not res_filter:
        return str(files[0])
    # parse "WxH"
    parts = res_filter.lower().split("x")
    if len(parts) != 2:
        return str(files[0])
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        return str(files[0])
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        meta = d.get("meta", d)
        if meta.get("width") == w and meta.get("height") == h:
            return str(f)
    return str(files[0])

b01_path = pick_file(b01_dir, b01_file, res_filter)
b02_path = pick_file(b02_dir, b02_file, res_filter)
b04_path = pick_file(b04_dir, b04_file, res_filter) if include_b04 else None

if not b01_path or not os.path.isfile(b01_path):
    print("错误: 未找到 B01 结果"); sys.exit(1)
if not b02_path or not os.path.isfile(b02_path):
    print("错误: 未找到 B02 结果"); sys.exit(1)
if include_b04 and (not b04_path or not os.path.isfile(b04_path)):
    print("错误: --all 已指定但未找到 B04 结果，请先运行 ./B04_multiple_gpu_4.sh"); sys.exit(1)

def get_metrics(d):
    if "metrics" in d:
        return d["metrics"], d.get("meta", {})
    return d, {}

def val(m, k, default=0):
    return m.get(k, default) or default

configs = [
    ("B01 1 GPU", b01_path),
    ("B02 2 GPU SP=2", b02_path),
]
if include_b04 and b04_path:
    configs.append(("B04 4 GPU SP=4", b04_path))

rows = []
for name, path in configs:
    with open(path) as f:
        d = json.load(f)
    m, meta = get_metrics(d)
    rows.append({
        "name": name,
        "path": path,
        "meta": meta,
        "lat_mean": val(m, "latency_mean"),
        "lat_p99": val(m, "latency_p99"),
        "lat_p95": val(m, "latency_p95"),
        "lat_p50": val(m, "latency_p50"),
        "thr": val(m, "throughput_qps"),
        "mem": val(m, "peak_memory_mb_max") / 1024,
        "completed": val(m, "completed_requests"),
    })

ref = rows[0]
header = "| 指标 | " + " | ".join(r["name"] for r in rows) + " |"
sep = "|:-----|" + "|".join(":---------:" for _ in rows) + "|"

def speedup(x, ref_val):
    if ref_val <= 0: return "-"
    return f"{ref_val/x:.2f}x"

lines = [
    "# B01 / B02 / B04 效率对比 (1 GPU vs 2 GPU vs 4 GPU)",
    "",
]
for r in rows:
    res = f"{r['meta'].get('width','?')}x{r['meta'].get('height','?')}"
    lines.append(f"**{r['name']}**: `{r['path']}` | 分辨率: {res} | 成功: {r['completed']}")
lines.append("")
lines.append("## Metrics 对比")
lines.append("")
lines.append(header)
lines.append(sep)
lines.append("| Latency Mean (s) | " + " | ".join(f"{r['lat_mean']:.4f}" for r in rows) + " |")
lines.append("| Latency P99 (s) | " + " | ".join(f"{r['lat_p99']:.4f}" for r in rows) + " |")
lines.append("| Latency P95 (s) | " + " | ".join(f"{r['lat_p95']:.4f}" for r in rows) + " |")
lines.append("| Latency P50 (s) | " + " | ".join(f"{r['lat_p50']:.4f}" for r in rows) + " |")
lines.append("| Throughput (req/s) | " + " | ".join(f"{r['thr']:.4f}" for r in rows) + " |")
lines.append("| Peak Memory (GB) | " + " | ".join(f"{r['mem']:.2f}" for r in rows) + " |")
lines.append("")
lines.append("## 相对 B01 的加速比 (Mean)")
lines.append("")
for i, r in enumerate(rows):
    if i == 0:
        lines.append(f"- {r['name']}: 基准")
    else:
        s = ref["lat_mean"] / r["lat_mean"] if r["lat_mean"] > 0 else 0
        lines.append(f"- {r['name']}: {s:.2f}x")
lines.append("")

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print("".join(lines))
print(f"\n✅ 对比报告已保存: {out_path}")
EOFPY
