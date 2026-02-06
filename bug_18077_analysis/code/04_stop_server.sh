#!/bin/bash
# 04: 停止 GLM-Image 服务器

set -e

PORT="${1:-30000}"

echo "=========================================="
echo "停止 GLM-Image 服务器（端口: $PORT）"
echo "=========================================="
echo ""

# 检查端口是否被占用
PID=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$PID" ]; then
    echo "✓ 端口 $PORT 未被占用，服务器可能已关闭"
    echo ""
    echo "验证："
    ps aux | grep "sglang serve" | grep -v grep || echo "  没有找到 sglang serve 进程"
    exit 0
fi

echo "找到进程 PID: $PID"
echo ""

# 显示进程信息
echo "进程信息："
ps -p $PID -o pid,cmd --no-headers || {
    echo "  进程已不存在"
    exit 0
}
echo ""

# 发送 SIGTERM 信号（优雅关闭）
echo "发送 SIGTERM 信号（优雅关闭）..."
kill -TERM $PID

# 等待进程退出
echo "等待进程退出（最多 30 秒）..."
for i in {1..30}; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "✓ 服务器已关闭"
        exit 0
    fi
    sleep 1
    echo -n "."
done
echo ""

# 如果还在运行，询问是否强制关闭
if kill -0 $PID 2>/dev/null; then
    echo ""
    echo "⚠️  服务器未在 30 秒内关闭"
    echo ""
    read -p "是否强制关闭？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "强制关闭进程..."
        kill -9 $PID
        sleep 1
        if ! kill -0 $PID 2>/dev/null; then
            echo "✓ 服务器已强制关闭"
        else
            echo "✗ 无法关闭进程，可能需要 root 权限"
            exit 1
        fi
    else
        echo "取消关闭，进程仍在运行"
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
