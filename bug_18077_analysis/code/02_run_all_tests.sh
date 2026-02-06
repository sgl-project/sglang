#!/bin/bash
# 02: 运行所有测试配置（一次性）
# 用法:
# ./02_run_all_tests.sh --backend sglang --port 30000 --dataset random \
#   --num-prompts 10 --widths 512,1024 --heights 512,1024 --max-concurrency 1,2 --request-rate inf \
#   --model zai-org/GLM-Image

set -e

# 中断处理：确保已完成的结果已保存
cleanup_on_interrupt() {
    echo ""
    echo "⚠️  检测到中断信号（Ctrl+C）"
    echo "✅ 已完成的结果已保存，可以安全退出"
    exit 130
}

# 注册信号处理
trap cleanup_on_interrupt SIGINT SIGTERM

# 中断处理函数：确保已完成的结果已保存
cleanup_on_interrupt() {
    echo ""
    echo "⚠️  检测到中断信号（Ctrl+C）"
    echo "✅ 已完成的结果已保存到: $RESULT_DIR"
    echo "   可以查看已完成的测试结果文件"
    exit 130
}

# 注册信号处理
trap cleanup_on_interrupt SIGINT SIGTERM

BACKEND="sglang"
PORT="30000"
NUM_PROMPTS="10"
DATASET="random"
WIDTH_LIST="512,1024"
HEIGHT_LIST="512,1024"
CONCURRENCY_LIST="1,2,4,8,16"
REQUEST_RATE="inf"
MODEL_NAME="zai-org/GLM-Image"
BASE_URL=""
RESULT_SUBDIR=""  # 可选：指定结果子目录，如 "multi_gpu" 或 "high_concurrency"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
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
        --request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --result-subdir)
            RESULT_SUBDIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法:"
            echo "  ./02_run_all_tests.sh --backend sglang|diffusers --port 30000 --dataset random \\"
            echo "    --num-prompts 10 --widths 512,1024 --heights 512,1024 --max-concurrency 1,2,4,8,16 --request-rate inf \\"
            echo "    --model zai-org/GLM-Image [--base-url http://localhost:30000] [--result-subdir multi_gpu]"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 确定结果目录
if [ -n "$RESULT_SUBDIR" ]; then
    RESULT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results_${RESULT_SUBDIR}"
else
    RESULT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results"
fi
mkdir -p "$RESULT_DIR"
echo "结果保存目录: $RESULT_DIR"

cd /data/users/yandache/workspaces/sglang/repo/sglang-src

echo "开始测试: $BACKEND 后端"
echo ""

echo "参数:"
echo "  后端: $BACKEND | 端口: $PORT | 请求数: $NUM_PROMPTS | 数据集: $DATASET"
if [ -n "$BASE_URL" ]; then
    EFFECTIVE_BASE_URL="$BASE_URL"
else
    EFFECTIVE_BASE_URL="http://localhost:$PORT"
fi
echo "  宽度: $WIDTH_LIST | 高度: $HEIGHT_LIST | 并发: $CONCURRENCY_LIST | 速率: $REQUEST_RATE"
echo "  模型: $MODEL_NAME | base_url: $EFFECTIVE_BASE_URL"
echo ""

to_list() {
    echo "${1//,/ }"
}

for width in $(to_list "$WIDTH_LIST"); do
    for height in $(to_list "$HEIGHT_LIST"); do
        for concurrency in $(to_list "$CONCURRENCY_LIST"); do
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            # 在文件名中包含 subdir 标识（如果有）
            if [ -n "$RESULT_SUBDIR" ]; then
                FILE_PREFIX="GLM-Image_${RESULT_SUBDIR}_${BACKEND}"
            else
                FILE_PREFIX="GLM-Image_${BACKEND}"
            fi
            OUTPUT_FILE="${RESULT_DIR}/${FILE_PREFIX}_w${width}_h${height}_n${NUM_PROMPTS}_c${concurrency}_${DATASET}_${TIMESTAMP}.json"
            TMP_METRICS="${OUTPUT_FILE%.json}.metrics.json"

            echo "测试: ${width}x${height}, 并发 ${concurrency}"
            python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
                --base-url "$EFFECTIVE_BASE_URL" \
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
        "base_url": "$EFFECTIVE_BASE_URL",
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
                echo ""
        done
    done
done

echo "✅ 所有测试完成"
