#!/bin/bash
# 05_clean_gpu: 清理 GPU 内存 / 进程，用于 A01/B01/B02 等测试前释放显存
# 用法:
#   ./05_clean_gpu.sh          - 默认全清理（杀所有占用 GPU 的进程）
#   ./05_clean_gpu.sh --sglang - 仅杀 sglang 相关进程 + 端口 30000
#   ./05_clean_gpu.sh --port N - 指定端口（默认 30000）

set -e

PORT=30000
FULL_CLEAN=true

while [ $# -gt 0 ]; do
    case "$1" in
        --sglang) FULL_CLEAN=false; shift ;;
        --all) FULL_CLEAN=true; shift ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "用法: $0 [--sglang] [--port N]"; exit 1 ;;
    esac
done

echo "=== 05_clean_gpu: 清理 GPU ==="
echo "端口: $PORT"
echo "模式: $([ "$FULL_CLEAN" = true ] && echo '全清理' || echo '仅 sglang')"

# 1. 停止端口上的服务
pid=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$pid" ]; then
    echo "停止端口 ${PORT} 上的进程: $pid"
    kill -TERM $pid 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 $pid 2>/dev/null; then break; fi
        sleep 1
    done
    kill -9 $pid 2>/dev/null || true
fi

# 2. 杀 sglang 相关进程
pkill -f "sglang serve" 2>/dev/null || true
pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' 2>/dev/null | xargs -r kill -9 2>/dev/null || true

sleep 2

# 3. --all: 杀所有占用 GPU 的进程
if [ "$FULL_CLEAN" = true ]; then
    echo "清理所有 GPU 进程..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    if command -v lsof >/dev/null 2>&1; then
        lsof /dev/nvidia* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9 2>/dev/null || true
    fi
    sleep 1
fi

echo "--- nvidia-smi ---"
nvidia-smi
echo "✅ 05_clean_gpu 完成"
