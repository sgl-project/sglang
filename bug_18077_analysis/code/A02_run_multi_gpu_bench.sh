#!/bin/bash
# A02: 多卡 benchmark，与 02 结构一致、测试量极少，仅用于与单卡 03 对比证明多卡效率
# 用法: 先运行 A01_start_server.sh，再在另一终端执行 ./A02_run_multi_gpu_bench.sh [--port 30000]
# 可选参数与 02 类似，默认仅跑 1 个配置（1024x1024 n3 c1），足以供 03 对比

set -e

BENCH_BASE="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark"
RESULTS_MULTI_DIR="${BENCH_BASE}/results_multi_gpu"
BENCH_DIR="${BENCH_BASE}/02112026"

BACKEND="sglang"
PORT="30000"
# 测试量急速降低：默认 3 个请求、1 个分辨率、1 个并发，仅够 03 对比
NUM_PROMPTS="${NUM_PROMPTS:-3}"
DATASET="${DATASET:-random}"
WIDTH_LIST="${WIDTH_LIST:-1024}"
HEIGHT_LIST="${HEIGHT_LIST:-1024}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MODEL_NAME="zai-org/GLM-Image"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --widths)
            WIDTH_LIST="$2"
            shift 2
            ;;
        --heights)
            HEIGHT_LIST="$2"
            shift 2
            ;;
        --max-concurrency)
            CONCURRENCY_LIST="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: ./A02_run_multi_gpu_bench.sh [--port 30000] [--num-prompts 3] [--widths 1024] [--heights 1024] [--max-concurrency 1] [--dataset random]"
            echo "默认极少测试量，仅用于与单卡 03 对比多卡效率。"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

BASE_URL="http://127.0.0.1:${PORT}"
mkdir -p "$RESULTS_MULTI_DIR" "$BENCH_DIR"
cd /data/users/yandache/workspaces/sglang
[ -f "env_sglang/bin/activate" ] && source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

echo "=== 多卡 SP Benchmark（与 02 同结构，测试量极少）==="
echo "base_url: $BASE_URL | 请求数: $NUM_PROMPTS | 数据集: $DATASET"
echo "宽度: $WIDTH_LIST | 高度: $HEIGHT_LIST | 并发: $CONCURRENCY_LIST"
echo "结果: $RESULTS_MULTI_DIR（供 03 对比） + $BENCH_DIR（双卡证明）"
echo ""

# 先检查服务器是否已就绪，避免卡在 "Waiting for service" 无提示
echo "检查服务器 $BASE_URL 是否就绪..."
if ! python3 -c "
import urllib.request
import sys
try:
    req = urllib.request.Request('$BASE_URL/health')
    urllib.request.urlopen(req, timeout=10)
    sys.exit(0)
except Exception as e:
    print(f'连接失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
    echo ""
    echo "错误: 无法连接 $BASE_URL（/health 无响应）。"
    echo "请先在同一台机器上运行 A01_start_server.sh，等终端里出现类似 \"Model is ready\" 或服务完全启动后，再运行本脚本。"
    echo "若 A01 正在加载模型，可等待几分钟后重新运行 A02。"
    exit 1
fi
echo "服务器已就绪，开始测试。"
echo ""

# 测试前 GPU 快照
TS_START=$(date +"%Y%m%d_%H%M%S")
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv > "${BENCH_DIR}/gpu_before_test_${TS_START}.txt" 2>&1 || true
nvidia-smi >> "${BENCH_DIR}/gpu_before_test_${TS_START}.txt" 2>&1 || true
echo "已保存: ${BENCH_DIR}/gpu_before_test_${TS_START}.txt"
echo ""

to_list() {
    echo "${1//,/ }"
}

for width in $(to_list "$WIDTH_LIST"); do
    for height in $(to_list "$HEIGHT_LIST"); do
        for concurrency in $(to_list "$CONCURRENCY_LIST"); do
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            FILE_PREFIX="GLM-Image_multi_gpu_${BACKEND}"
            OUTPUT_FILE="${RESULTS_MULTI_DIR}/${FILE_PREFIX}_w${width}_h${height}_n${NUM_PROMPTS}_c${concurrency}_${DATASET}_${TIMESTAMP}.json"
            TMP_METRICS="${OUTPUT_FILE%.json}.metrics.json"

            echo "测试: ${width}x${height}, 并发 ${concurrency}"
            python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
                --base-url "$BASE_URL" \
                --dataset "$DATASET" \
                --num-prompts "$NUM_PROMPTS" \
                --max-concurrency "$concurrency" \
                --request-rate "$REQUEST_RATE" \
                --width "$width" \
                --height "$height" \
                --output-file "$TMP_METRICS"

            python3 - << EOF
import json
with open("$TMP_METRICS", "r") as f:
    metrics = json.load(f)
payload = {
    "meta": {
        "model": "$MODEL_NAME",
        "backend": "$BACKEND",
        "base_url": "$BASE_URL",
        "dataset": "$DATASET",
        "num_prompts": int("$NUM_PROMPTS"),
        "width": int("$width"),
        "height": int("$height"),
        "concurrency": int("$concurrency"),
        "request_rate": "$REQUEST_RATE",
        "timestamp": "$TIMESTAMP",
    },
    "metrics": metrics,
}
with open("$OUTPUT_FILE", "w") as f:
    json.dump(payload, f, indent=2)
EOF
            rm -f "$TMP_METRICS"
            echo "结果: $OUTPUT_FILE"
            cp "$OUTPUT_FILE" "${BENCH_DIR}/multi_gpu_result_w${width}_h${height}_n${NUM_PROMPTS}_c${concurrency}_${TIMESTAMP}.json"
            echo ""
        done
    done
done

# 测试后 GPU 快照（证明推理时两卡都在工作）
TS_END=$(date +"%Y%m%d_%H%M%S")
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv > "${BENCH_DIR}/gpu_after_test_${TS_END}.txt" 2>&1 || true
nvidia-smi >> "${BENCH_DIR}/gpu_after_test_${TS_END}.txt" 2>&1 || true
echo "已保存: ${BENCH_DIR}/gpu_after_test_${TS_END}.txt"
echo "✅ 多卡测试完成。用 ./03_compare_single_vs_multi_gpu.sh 与单卡结果对比效率。"
