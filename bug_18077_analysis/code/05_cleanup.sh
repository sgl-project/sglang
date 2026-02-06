#!/bin/bash
# 05: 清理环境，为下一个测试做准备
# 用法: ./05_cleanup.sh [port]

set -e

PORT="${1:-30000}"

echo "=========================================="
echo "清理环境（端口: $PORT）"
echo "=========================================="
echo ""

# 1. 停止服务器（如果还在运行）
echo "1️⃣  检查并停止服务器..."
if [ -f "./04_stop_server.sh" ]; then
    ./04_stop_server.sh "$PORT" || true
else
    PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "   找到进程 $PID，正在停止..."
        kill -TERM "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
fi
echo "   ✓ 服务器已停止"
echo ""

# 2. 清理 GPU 缓存
echo "2️⃣  清理 GPU 缓存..."
# 检查是否有 Python 进程占用 GPU
PYTHON_GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" || true)
if [ -n "$PYTHON_GPU_PROCS" ]; then
    echo "   发现 GPU 进程，正在清理..."
    echo "$PYTHON_GPU_PROCS" | while read pid; do
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
fi

# 清理 PyTorch CUDA 缓存（通过 Python）
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("   ✓ PyTorch CUDA 缓存已清理")
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
        print(f"   ✓ GPU {i} 内存统计已重置")
else:
    print("   ⚠️  未检测到 CUDA")
EOF
echo ""

# 3. 清理 Python 缓存
echo "3️⃣  清理 Python 缓存..."
# 清理 __pycache__
find /data/users/yandache/workspaces/sglang/repo/sglang-src/python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /data/users/yandache/workspaces/sglang/repo/sglang-src/python -name "*.pyc" -delete 2>/dev/null || true
echo "   ✓ Python 缓存已清理"
echo ""

# 4. 清理临时文件
echo "4️⃣  清理临时文件..."
# 清理测试脚本产生的临时 metrics 文件
rm -f /data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results/*.metrics.json 2>/dev/null || true
echo "   ✓ 临时文件已清理"
echo ""

# 5. 验证 GPU 状态
echo "5️⃣  验证 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name mem_used mem_total util; do
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    util=$(echo "$util" | xargs)
    if [ "$mem_used" -lt 100 ] && [ "$util" -eq 0 ]; then
        echo "   ✓ GPU $idx ($name): ${mem_used}MB/${mem_total}MB, 利用率 ${util}%"
    else
        echo "   ⚠️  GPU $idx ($name): ${mem_used}MB/${mem_total}MB, 利用率 ${util}%"
    fi
done
echo ""

# 6. 检查端口
echo "6️⃣  检查端口占用..."
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "   ⚠️  端口 $PORT 仍被占用"
    lsof -ti:$PORT | xargs ps -p
else
    echo "   ✓ 端口 $PORT 空闲"
fi
echo ""

# 7. 检查进程
echo "7️⃣  检查残留进程..."
SGLANG_PROCS=$(ps aux | grep -E "sglang|python.*serve|uvicorn" | grep -v grep || true)
if [ -z "$SGLANG_PROCS" ]; then
    echo "   ✓ 没有残留进程"
else
    echo "   ⚠️  发现残留进程："
    echo "$SGLANG_PROCS"
fi
echo ""

echo "=========================================="
echo "✅ 清理完成"
echo "=========================================="
echo ""
echo "环境已准备好，可以开始新的测试了！"
echo ""
