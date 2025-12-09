#!/bin/bash

# 快速 pprof 分析脚本
# 收集 30 秒 CPU profile 并立即显示 top 结果

set -e

PPROF_PORT=${PPROF_PORT:-6060}
DURATION=${DURATION:-30}

echo "=========================================="
echo "快速 pprof 分析"
echo "=========================================="
echo "PPROF_PORT: $PPROF_PORT"
echo "DURATION: ${DURATION}s"
echo ""
echo "提示: 在收集数据期间，请向服务器发送请求"
echo "      可以使用: ./pprof_test.sh"
echo ""

# 检查 pprof 是否可用
if ! curl -s "http://localhost:${PPROF_PORT}/debug/pprof/" > /dev/null 2>&1; then
    echo "错误: pprof 未启用。请设置环境变量："
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    exit 1
fi

echo "开始收集 CPU Profile (${DURATION}秒)..."
echo ""

# 收集 CPU profile 并直接显示 top 结果
go tool pprof -top -cum "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=${DURATION}"

echo ""
echo "=========================================="
echo "分析完成"
echo "=========================================="
echo ""
echo "更多分析选项："
echo "  # 交互式查看"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30"
echo ""
echo "  # 查看堆内存"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/heap"
echo ""
echo "  # 查看 goroutine"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/goroutine"
echo ""
echo "  # 生成 Web UI"
echo "  go tool pprof -http=:8080 http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30"
echo ""


