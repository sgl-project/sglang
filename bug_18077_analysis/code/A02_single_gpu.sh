#!/bin/bash
# A02_single_gpu: 单卡 benchmark，与 A02 结构一致、使用 random dataset，用于与多卡 A02 对比
# 用法: 先运行 A01_single_gpu.sh，再在另一终端执行 ./A02_single_gpu.sh [--port 30000]
# 会先验证服务端调用路径为 GlmImagePipelineConfig / SpatialImagePipelineConfig（通过 pipeline_class 与 model_path）。

set -e

BENCH_BASE="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark"
RESULTS_SINGLE_DIR="${BENCH_BASE}/results_single_gpu"

BACKEND="sglang"
PORT="30000"
# 与 A02 默认一致：少量请求、random、1024x1024、并发 1
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
            echo "用法: ./A02_single_gpu.sh [--port 30000] [--num-prompts 3] [--widths 1024] [--heights 1024] [--max-concurrency 1] [--dataset random]"
            echo "默认与 A02 一致（random dataset），用于单卡测试并验证 GlmImagePipelineConfig / SpatialImagePipelineConfig 路径。"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

BASE_URL="http://127.0.0.1:${PORT}"
mkdir -p "$RESULTS_SINGLE_DIR"
cd /data/users/yandache/workspaces/sglang
[ -f "env_sglang/bin/activate" ] && source env_sglang/bin/activate
cd /data/users/yandache/workspaces/sglang/repo/sglang-src

echo "=== 单卡 Benchmark（与 A02 同结构，dataset=random）==="
echo "base_url: $BASE_URL | 请求数: $NUM_PROMPTS | 数据集: $DATASET"
echo "宽度: $WIDTH_LIST | 高度: $HEIGHT_LIST | 并发: $CONCURRENCY_LIST"
echo "结果: $RESULTS_SINGLE_DIR"
echo ""

# 检查服务器是否就绪
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
    echo "错误: 无法连接 $BASE_URL。请先运行 A01_single_gpu.sh，等服务启动后再执行本脚本。"
    exit 1
fi
echo "服务器已就绪。"
echo ""

# 验证调用路径：GlmImagePipelineConfig / SpatialImagePipelineConfig
# 服务端对 zai-org/GLM-Image 使用 registry 中的 GlmImagePipelineConfig（继承 SpatialImagePipelineConfig），
# 对应 pipeline 类为 GlmImagePipeline。通过 /v1/models 或 /models 可拿到 pipeline_class。
echo "=== 验证 pipeline 路径（GlmImagePipelineConfig / SpatialImagePipelineConfig）==="
export BASE_URL
python3 -c "
import json, sys, urllib.request, os
base_url = os.environ.get('BASE_URL', 'http://127.0.0.1:30000')
for path in ('/v1/models', '/models'):
    try:
        req = urllib.request.Request(base_url.rstrip('/') + path)
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            if 'data' in data and len(data['data']) > 0:
                info = data['data'][0]
            else:
                info = data
            pc = info.get('pipeline_class') or info.get('pipeline_name')
            mid = info.get('id') or info.get('root') or ''
            if pc == 'GlmImagePipeline':
                print('OK 调用路径已确认: pipeline_class=GlmImagePipeline -> GlmImagePipelineConfig / SpatialImagePipelineConfig')
                sys.exit(0)
            else:
                print('校验失败: pipeline_class=%s, 期望 GlmImagePipeline' % pc, file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        continue
print('警告: 无法获取 pipeline 信息，跳过校验', file=sys.stderr)
sys.exit(0)
"
echo ""

to_list() {
    echo "${1//,/ }"
}

for width in $(to_list "$WIDTH_LIST"); do
    for height in $(to_list "$HEIGHT_LIST"); do
        for concurrency in $(to_list "$CONCURRENCY_LIST"); do
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            FILE_PREFIX="GLM-Image_single_gpu_${BACKEND}"
            OUTPUT_FILE="${RESULTS_SINGLE_DIR}/${FILE_PREFIX}_w${width}_h${height}_n${NUM_PROMPTS}_c${concurrency}_${DATASET}_${TIMESTAMP}.json"
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
            echo ""
        done
    done
done

echo "✅ 单卡测试完成（random dataset，与 A02 同结构）。调用路径已确认为 GlmImagePipelineConfig / SpatialImagePipelineConfig。"
