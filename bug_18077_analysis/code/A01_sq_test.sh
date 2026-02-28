#!/bin/bash
# A01_sq_test.sh — 单卡→多卡验证 + 性能取证（证明新 SP 比单卡更快）
# 用法: ./A01_sq_test.sh
# 依赖: bug_18077_analysis/code 下 04_stop_server.sh（或自动 kill 端口进程）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_BASE="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark"
RESULTS_DIR="${BENCH_BASE}/results"
RESULTS_MULTI_DIR="${BENCH_BASE}/results_multi_gpu"
PORT=30000
MODEL="zai-org/GLM-Image"
# 高分辨率下跑少量请求即可证明 speedup
PERF_NUM_PROMPTS=6
PERF_WIDTH=1024
PERF_HEIGHT=1024
PERF_CONCURRENCY=1

# ---------- 环境 ----------
setup_env() {
    if [ -f "/data/users/yandache/_shared/tools/env.sh" ]; then
        source /data/users/yandache/_shared/tools/env.sh
        echo "✓ 已加载 _shared/tools/env.sh"
    else
        export SPACE=/data/users/yandache
        export HF_HOME="$SPACE/_shared/cache/hf"
        export TRANSFORMERS_CACHE="$SPACE/_shared/cache/hf/transformers"
        export XDG_CACHE_HOME="$SPACE/_shared/cache/xdg/cache"
        export TMPDIR="$SPACE/tmp"
    fi
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TMPDIR"
}

# ---------- 工作目录与 Python 环境 ----------
cd /data/users/yandache/workspaces/sglang
if [ -f "env_sglang/bin/activate" ]; then
    source env_sglang/bin/activate
    echo "✓ 已激活 env_sglang"
fi
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

# ---------- 等待服务就绪（先等端口监听，再多等几秒让模型加载完） ----------
wait_for_server() {
    local port="${1:-30000}"
    local max_wait="${2:-180}"
    local waited=0
    echo "等待服务 localhost:${port} 就绪（最多 ${max_wait}s）..."
    while [ "$waited" -lt "$max_wait" ]; do
        if python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(2); s.connect(('127.0.0.1',$port)); s.close()" 2>/dev/null; then
            echo "✓ 端口已监听（${waited}s），再等 15s 让模型就绪..."
            sleep 15
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "✗ 超时：服务在 ${max_wait}s 内未就绪"
    return 1
}

# ---------- 停止端口上的服务 ----------
stop_server() {
    local port="${1:-30000}"
    local pid
    pid=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "停止端口 ${port} 上的进程: $pid"
        kill -TERM $pid 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 $pid 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    fi
    pkill -f "sglang serve" 2>/dev/null || true
    sleep 2
}

# ---------- 运行一次小规模 bench（用于 sanity） ----------
run_sanity_bench() {
    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
        --model "$MODEL" \
        --dataset random \
        --num-prompts 2 \
        --width 512 \
        --height 512 \
        --max-concurrency 1 \
        --base-url "http://127.0.0.1:${PORT}"
}

# ---------- 运行性能 bench 并写入指定目录 ----------
run_perf_bench() {
    local out_dir="$1"
    local subdir_label="$2"
    mkdir -p "$out_dir"
    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local prefix="GLM-Image"
    [ -n "$subdir_label" ] && prefix="GLM-Image_${subdir_label}"
    local out_file="${out_dir}/${prefix}_w${PERF_WIDTH}_h${PERF_HEIGHT}_n${PERF_NUM_PROMPTS}_c${PERF_CONCURRENCY}_random_${ts}.json"
    local tmp_metrics="${out_file%.json}.metrics.json"

    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset random \
        --num-prompts "$PERF_NUM_PROMPTS" \
        --max-concurrency "$PERF_CONCURRENCY" \
        --width "$PERF_WIDTH" \
        --height "$PERF_HEIGHT" \
        --output-file "$tmp_metrics"

    python3 - << EOF
import json
with open("$tmp_metrics", "r") as f:
    metrics = json.load(f)
payload = {
    "meta": {
        "model": "$MODEL",
        "backend": "sglang",
        "base_url": "http://127.0.0.1:$PORT",
        "dataset": "random",
        "num_prompts": $PERF_NUM_PROMPTS,
        "width": $PERF_WIDTH,
        "height": $PERF_HEIGHT,
        "concurrency": $PERF_CONCURRENCY,
        "timestamp": "$ts",
    },
    "metrics": metrics,
}
with open("$out_file", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    rm -f "$tmp_metrics"
    echo "结果: $out_file"
}

# ---------- 生成单卡 vs 多卡对比报告 ----------
write_comparison() {
    local single_dir="$1"
    local multi_dir="$2"
    local out_md="$3"
    python3 - "$single_dir" "$multi_dir" "$out_md" << 'EOFPY'
import json, re, sys
from pathlib import Path

single_dir, multi_dir, out_md = sys.argv[1], sys.argv[2], sys.argv[3]

def load_metrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "metrics" in data:
        return data.get("meta", {}), data["metrics"]
    return {}, data

def config_from_name(name):
    m = re.search(r'w(\d+)_h(\d+)_n(\d+)_c(\d+)_', name)
    if m:
        return f"w{m.group(1)}_h{m.group(2)}_c{m.group(4)}"
    return None

def latest_file(files):
    return max(files, key=lambda f: f.stat().st_mtime) if files else None

single_dir = Path(single_dir)
multi_dir = Path(multi_dir)
single_by_key = {}
multi_by_key = {}
for f in single_dir.glob("*sglang*.json"):
    k = config_from_name(f.name)
    if k and (k not in single_by_key or f.stat().st_mtime > single_by_key[k].stat().st_mtime):
        single_by_key[k] = f
for f in multi_dir.glob("*sglang*.json"):
    k = config_from_name(f.name)
    if k and (k not in multi_by_key or f.stat().st_mtime > multi_by_key[k].stat().st_mtime):
        multi_by_key[k] = f

common = set(single_by_key) & set(multi_by_key)
if not common:
    print("未找到可对比的 single/multi 结果")
    sys.exit(1)

with open(out_md, 'w', encoding='utf-8') as f:
    f.write("# 单卡 vs 多卡 SP 性能对比 (A01_sq_test)\n\n")
    for key in sorted(common):
        s_meta, s_m = load_metrics(str(single_by_key[key]))
        m_meta, m_m = load_metrics(str(multi_by_key[key]))
        lat_s, lat_m = s_m.get("latency_mean", 0), m_m.get("latency_mean", 0)
        thr_s, thr_m = s_m.get("throughput_qps", 0), m_m.get("throughput_qps", 0)
        speedup = lat_s / lat_m if lat_m > 0 else 0
        f.write(f"## {key}\n\n")
        f.write(f"| 指标 | 单卡 | 多卡 SP | 变化 |\n")
        f.write("|:-----|:-----|:--------|:-----|\n")
        f.write(f"| 延迟 (平均) | {lat_s:.2f}s | {lat_m:.2f}s | **{speedup:.2f}x faster** |\n")
        f.write(f"| 吞吐量 | {thr_s:.4f} | {thr_m:.4f} | {thr_m/thr_s:.2f}x |\n\n")
print(f"对比报告: {out_md}")
EOFPY
}

# ========== 主流程 ==========
setup_env
mkdir -p "$RESULTS_DIR" "$RESULTS_MULTI_DIR"

echo ""
echo "=============================================="
echo "  Phase 1: 单卡功能验证 (Sanity Check)"
echo "=============================================="
stop_server "$PORT"
export CUDA_VISIBLE_DEVICES=0
(
    sglang serve --model-path "$MODEL" --backend sglang --port "$PORT" --trust-remote-code
) &
SRV_PID=$!
if ! wait_for_server "$PORT" 180; then
    kill -9 $SRV_PID 2>/dev/null || true
    exit 1
fi
run_sanity_bench
stop_server "$PORT"
echo "Phase 1 完成：单卡逻辑正常"
echo ""

echo "=============================================="
echo "  Phase 2: 2 GPU 序列并行验证 (SP 修复点)"
echo "=============================================="
stop_server "$PORT"
export CUDA_VISIBLE_DEVICES=0,1
(
    sglang serve --model-path "$MODEL" --backend sglang --port "$PORT" --trust-remote-code \
        --tp 2 --sp-degree 2 --ulysses-degree 2
) &
SRV_PID=$!
if ! wait_for_server "$PORT" 200; then
    kill -9 $SRV_PID 2>/dev/null || true
    echo "Phase 2 失败：2 GPU SP 未就绪（若日志有 einops 报错则说明需再检查基类）"
    exit 1
fi
run_sanity_bench
stop_server "$PORT"
echo "Phase 2 完成：2 GPU SP 无 einops 报错，Model is ready"
echo ""

echo "=============================================="
echo "  Phase 3: 单卡性能 Baseline (${PERF_WIDTH}x${PERF_HEIGHT})"
echo "=============================================="
stop_server "$PORT"
export CUDA_VISIBLE_DEVICES=0
(
    sglang serve --model-path "$MODEL" --backend sglang --port "$PORT" --trust-remote-code
) &
SRV_PID=$!
if ! wait_for_server "$PORT" 180; then
    kill -9 $SRV_PID 2>/dev/null || true
    exit 1
fi
run_perf_bench "$RESULTS_DIR" "sglang"
stop_server "$PORT"
echo "Phase 3 完成：单卡结果已写入 results/"
echo ""

echo "=============================================="
echo "  Phase 4: 2 卡 SP 性能 (${PERF_WIDTH}x${PERF_HEIGHT})"
echo "=============================================="
stop_server "$PORT"
export CUDA_VISIBLE_DEVICES=0,1
(
    sglang serve --model-path "$MODEL" --backend sglang --port "$PORT" --trust-remote-code \
        --tp 2 --sp-degree 2 --ulysses-degree 2
) &
SRV_PID=$!
if ! wait_for_server "$PORT" 200; then
    kill -9 $SRV_PID 2>/dev/null || true
    exit 1
fi
run_perf_bench "$RESULTS_MULTI_DIR" "multi_gpu_sglang"
stop_server "$PORT"
echo "Phase 4 完成：多卡结果已写入 results_multi_gpu/"
echo ""

echo "=============================================="
echo "  Phase 5: 生成单卡 vs 多卡对比报告"
echo "=============================================="
TS=$(date +"%Y%m%d_%H%M%S")
COMPARE_MD="${RESULTS_MULTI_DIR}/comparison_single_vs_multi_gpu_${TS}.md"
write_comparison "$RESULTS_DIR" "$RESULTS_MULTI_DIR" "$COMPARE_MD"
echo ""

echo "=============================================="
echo "  A01_sq_test 全部完成"
echo "=============================================="
echo "单卡结果: $RESULTS_DIR"
echo "多卡结果: $RESULTS_MULTI_DIR"
echo "对比报告: $COMPARE_MD"
echo ""
echo "在另一终端可用以下命令监控 GPU："
echo "  watch -n 1 nvidia-smi"
echo ""
