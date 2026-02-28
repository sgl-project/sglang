#!/bin/bash
# B04_multiple_gpu_4: 4 卡 SP=4 效率测试（server + test 一体）
# 用法: ./B04_multiple_gpu_4.sh [--port 30000] [--num-prompts 10] [--stress|--stress-max] [--width W] [--height H]
#   --stress:     1024x1024
#   --stress-max: 1536x1536

set -e

PORT="30000"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
DATASET="${DATASET:-random}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MODEL_NAME="zai-org/GLM-Image"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --stress) WIDTH="1024"; HEIGHT="1024"; shift ;;
        --stress-max) WIDTH="1536"; HEIGHT="1536"; shift ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        -h|--help)
            echo "用法: ./B04_multiple_gpu_4.sh [--port 30000] [--num-prompts N] [--stress|--stress-max] [--width W] [--height H]"
            echo "  4 GPU, tp=1, sp=4"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 1. 环境变量
if [ -f "/data/users/yandache/_shared/tools/env.sh" ]; then
    source /data/users/yandache/_shared/tools/env.sh
    echo "✓ 已加载环境变量配置 (_shared/tools/env.sh)"
else
    export SPACE=/data/users/yandache
    export HF_HOME="$SPACE/_shared/cache/hf"
    export TRANSFORMERS_CACHE="$SPACE/_shared/cache/hf/transformers"
    export XDG_CACHE_HOME="$SPACE/_shared/cache/xdg/cache"
    export TMPDIR="$SPACE/tmp"
fi
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

# 2. 项目目录与虚拟环境
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

# 3. 检查 diffusion
if ! python3 -c "import diffusers" 2>/dev/null; then
    echo "错误: 未安装 diffusers，请 pip install -e \".[diffusion]\""
    exit 1
fi

BASE_URL="http://127.0.0.1:${PORT}"
OUT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/B04_multiple_gpu_4"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUT_DIR}/run_${TIMESTAMP}.log"
RESULT_FILE="${OUT_DIR}/result_${TIMESTAMP}.json"
TMP_METRICS="${OUT_DIR}/metrics_tmp_${TIMESTAMP}.json"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo ">>> 停止 server (PID=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== B04 4 卡 SP=4 效率测试 $TIMESTAMP ==="
echo "BASE_URL=$BASE_URL"
echo "请求数: $NUM_PROMPTS | 并发: $CONCURRENCY | 尺寸: ${WIDTH}x${HEIGHT}"
echo "日志文件: $LOG_FILE"
echo ""

# 4. 4 卡可见
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 5. 启动服务（tp=1, sp=4）
echo "--- Step 1: 启动 4 卡 SP=4 server (tp=1, sp=4) ---"
sglang serve \
  --model-path zai-org/GLM-Image \
  --num-gpus 4 \
  --tp-size 1 \
  --sp-degree 4 \
  --port "$PORT" \
  --trust-remote-code &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo ""

# 6. 等待服务就绪
echo "--- Step 2: 等待 health 就绪 ---"
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" 2>/dev/null | grep -q 200; then
        echo "health: OK (等待 ${ELAPSED}s)"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "错误: 等待 health 超时 (${MAX_WAIT}s)"
    exit 1
fi
echo ""

# 7. 验证 pipeline
echo "--- Step 3: 验证 pipeline ---"
MODEL_INFO=$(curl -s "$BASE_URL/v1/models" 2>/dev/null || curl -s "$BASE_URL/models" 2>/dev/null || true)
if echo "$MODEL_INFO" | grep -q "GlmImagePipeline"; then
    echo "pipeline_class: GlmImagePipeline OK"
else
    echo "pipeline 信息: $MODEL_INFO"
fi
echo ""

# 8. bench_serving
echo "--- Step 4: bench_serving 多请求 (${NUM_PROMPTS} 次, ${WIDTH}x${HEIGHT}) ---"
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --base-url "$BASE_URL" \
    --dataset "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate "$REQUEST_RATE" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --output-file "$TMP_METRICS" \
    --model "$MODEL_NAME"

# 9. 写入结果 JSON
python3 - "$RESULT_FILE" "$TMP_METRICS" "$TIMESTAMP" "$WIDTH" "$HEIGHT" << 'EOFPY'
import json, sys, os
_, result_file, tmp_metrics, timestamp, width, height = sys.argv
with open(tmp_metrics, "r") as f:
    metrics = json.load(f)
d = {
    "meta": {
        "config": "B04_multiple_gpu_4",
        "gpus": 4,
        "tp": 1,
        "sp": 4,
        "timestamp": timestamp,
        "model": "zai-org/GLM-Image",
        "width": int(width),
        "height": int(height),
    },
    "metrics": metrics,
}
with open(result_file, "w") as f:
    json.dump(d, f, indent=2)
try:
    os.remove(tmp_metrics)
except OSError:
    pass
EOFPY

echo "--- B04 4 卡测试完成 ---"
echo "结果文件: $RESULT_FILE"
echo "完整日志: $LOG_FILE"
echo ""
