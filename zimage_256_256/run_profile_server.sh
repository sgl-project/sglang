#!/bin/bash
# ============================================================
# Z-Image-Turbo nsys Profiling — Server + Client 一键脚本
#
# 用法:
#   bash run_profile_server.sh [256|512|1024] [--no-nsys]
#
# 示例:
#   bash run_profile_server.sh 256           # 用 nsys profile 启动
#   bash run_profile_server.sh 256 --no-nsys # 不用 nsys（测 E2E 基准时间）
# ============================================================

set -euo pipefail

# ============================================================
# 配置
# ============================================================
IMAGE_SIZE="${1:-256}"
NO_NSYS="${2:-}"

MODEL="/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo"
FP8_TRANSFORMER="$MODEL/transformer-FP8-block128"
PORT=30000
HOST="127.0.0.1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLIENT_SCRIPT="$SCRIPT_DIR/profile_client.py"

# nsys 输出目录
NSYS_OUT_DIR="$SCRIPT_DIR/zimage_bench/nsys_server"
mkdir -p "$NSYS_OUT_DIR"

# 生成图片保存目录
IMG_OUT_DIR="$SCRIPT_DIR/profile_images/${IMAGE_SIZE}x${IMAGE_SIZE}"
mkdir -p "$IMG_OUT_DIR"

echo "============================================================"
echo " Z-Image-Turbo Profiling"
echo " Resolution:  ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo " Model:       $MODEL"
echo " Port:        $PORT"
echo " nsys:        $([ "$NO_NSYS" = "--no-nsys" ] && echo "OFF (baseline E2E)" || echo "ON")"
echo "============================================================"
echo ""

# ============================================================
# 清理函数：确保退出时杀掉 server
# ============================================================
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ""
        echo "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup EXIT

# ============================================================
# 构建 server 命令
# ============================================================
# BF16 baseline server 命令
SERVER_CMD=(
    sglang serve
    --model-path "$MODEL"
    --port "$PORT"
    --host "$HOST"
    --warmup
    --text-encoder-precisions bf16
)

# FP8 server 命令（取消注释以使用）
SERVER_CMD=(
    sglang serve
    --model-path "$MODEL"
    --transformer-weights-path "$FP8_TRANSFORMER"
    --port "$PORT"
    --host "$HOST"
    --warmup
    --text-encoder-precisions bf16
)

# ============================================================
# 启动 Server
# ============================================================
# 记录脚本启动时间戳（epoch 秒，含小数），用于计算 server 启动耗时
SCRIPT_START_TIME=$(python3 -c "import time; print(f'{time.time():.3f}')")
export SCRIPT_START_TIME

if [ "$NO_NSYS" = "--no-nsys" ]; then
    echo "[1/3] Starting server WITHOUT nsys (baseline E2E measurement)..."
    "${SERVER_CMD[@]}" &
    SERVER_PID=$!
else
    echo "[1/3] Starting server WITH nsys profile..."
    NSYS_OUTPUT="${NSYS_OUT_DIR}/server_bf16_fp8_deepgemm_${IMAGE_SIZE}x${IMAGE_SIZE}"
    # --delay: 给 server 足够启动时间（含 warmup）
    # --duration: 足够长以覆盖所有 client 请求
    CUDA_VISIBLE_DEVICES=0 nsys profile \
        --trace=cuda,nvtx,cudnn,cublas \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --delay=40 \
        --duration=20 \
        -o "$NSYS_OUTPUT" \
        "${SERVER_CMD[@]}" &
    SERVER_PID=$!
    echo "  nsys output: ${NSYS_OUTPUT}.nsys-rep"
fi
echo "  Server PID: $SERVER_PID"

# ============================================================
# 等待 Server 就绪 + 运行 Client
# ============================================================
echo ""
echo "[2/3] Waiting for server to be ready..."

python3 "$CLIENT_SCRIPT" \
    --host "$HOST" \
    --port "$PORT" \
    --size "$IMAGE_SIZE" \
    --seed 42 \
    --num-prompts 10 \
    --warmup 2 \
    --save-images \
    --output-dir "$IMG_OUT_DIR" \
    --wait-server \
    --wait-timeout 300

# ============================================================
# 等待 nsys 完成并写出 .nsys-rep
# ============================================================
if [ "$NO_NSYS" != "--no-nsys" ]; then
    echo ""
    echo "[3/3] Client done. Waiting for nsys to finish (delay+duration=${NSYS_DELAY}+${NSYS_DURATION}s)..."
    echo "  nsys will terminate the server when profiling completes."
    # 等待 nsys 进程（即 SERVER_PID）自然结束
    # 取消 trap 以避免 cleanup 提前 kill nsys
    trap - EXIT
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo "  nsys finished."
    if [ -f "${NSYS_OUTPUT}.nsys-rep" ]; then
        echo "  ✅ Output: ${NSYS_OUTPUT}.nsys-rep"
    else
        echo "  ❌ WARNING: ${NSYS_OUTPUT}.nsys-rep not found!"
    fi
else
    echo ""
    echo "[3/3] Done."
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "下一步："
if [ "$NO_NSYS" = "--no-nsys" ]; then
    echo "  记录上面输出的 elapsed_nonprofiled_sec 值"
    echo "  然后用 nsys 模式重新运行一次："
    echo "    bash run_profile_server.sh $IMAGE_SIZE"
else
    echo "  1. 不用 nsys 再运行一次获取 baseline E2E："
    echo "     bash run_profile_server.sh $IMAGE_SIZE --no-nsys"
    echo ""
    echo "  2. 用 gputrc2graph.py 分析："
    echo "     python3 gputrc2graph.py \\"
    echo "       --in_file ${NSYS_OUTPUT}.nsys-rep,sglang,zimage,<E2E_SEC> \\"
    echo "       --out_dir results/ --title 'ZImage BF16 ${IMAGE_SIZE}x${IMAGE_SIZE}'"
fi
