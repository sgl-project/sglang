#!/bin/bash

# pprof 性能分析脚本
# 用于分析 Go OpenAI 服务器的性能瓶颈

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
PPROF_PORT=${PPROF_PORT:-6060}
SERVER_PORT=${SERVER_PORT:-8080}
DURATION=${DURATION:-60}  # 性能测试持续时间（秒）
OUTPUT_DIR="./pprof_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "pprof 性能分析工具"
echo "=========================================="
echo "PPROF_PORT: $PPROF_PORT"
echo "SERVER_PORT: $SERVER_PORT"
echo "DURATION: ${DURATION}s"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# 检查 go tool pprof 是否可用
if ! command -v go &> /dev/null; then
    echo "错误: 未找到 go 命令"
    exit 1
fi

# 检查服务器是否在运行
check_server() {
    if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 检查 pprof 是否可用
check_pprof() {
    if curl -s "http://localhost:${PPROF_PORT}/debug/pprof/" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 启动服务器（如果未运行）
if ! check_server; then
    echo "服务器未运行，请先启动服务器："
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    echo "  ./oai_server"
    echo ""
    echo "或者使用以下命令启动："
    echo "  PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT ./oai_server"
    echo ""
    read -p "是否现在启动服务器？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "启动服务器..."
        PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT ./oai_server &
        SERVER_PID=$!
        echo "服务器 PID: $SERVER_PID"
        
        # 等待服务器启动
        echo "等待服务器启动..."
        for i in {1..30}; do
            if check_server; then
                echo "服务器已启动"
                break
            fi
            sleep 1
        done
        
        if ! check_server; then
            echo "错误: 服务器启动失败"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    else
        exit 1
    fi
fi

# 检查 pprof 是否可用
if ! check_pprof; then
    echo "错误: pprof 未启用。请设置环境变量："
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    exit 1
fi

echo "开始收集性能数据..."
echo ""

# 1. CPU Profile (30秒)
echo "[1/6] 收集 CPU Profile (30秒)..."
go tool pprof -proto -output="$OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30" &
CPU_PID=$!

# 2. 同时收集 Heap Profile
echo "[2/6] 收集 Heap Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" &
HEAP_PID=$!

# 3. 收集 Goroutine Profile
echo "[3/6] 收集 Goroutine Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/goroutine_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/goroutine" &
GOROUTINE_PID=$!

# 4. 收集 Mutex Profile
echo "[4/6] 收集 Mutex Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/mutex_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/mutex" &
MUTEX_PID=$!

# 5. 收集 Block Profile
echo "[5/6] 收集 Block Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/block_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/block" &
BLOCK_PID=$!

# 6. 运行性能测试（在收集 CPU profile 期间）
echo "[6/6] 运行性能测试..."
echo "提示: 请使用您的性能测试工具（如 curl、ab、wrk 等）向服务器发送请求"
echo "      CPU profile 将收集 30 秒的性能数据"
echo ""

# 等待 CPU profile 完成
wait $CPU_PID
echo "CPU Profile 收集完成"

# 等待其他 profiles
wait $HEAP_PID
wait $GOROUTINE_PID
wait $MUTEX_PID
wait $BLOCK_PID

echo ""
echo "=========================================="
echo "性能数据收集完成！"
echo "=========================================="
echo ""
echo "生成的分析文件："
ls -lh "$OUTPUT_DIR"/*_${TIMESTAMP}.* 2>/dev/null || true
echo ""

# 生成分析报告
echo "生成分析报告..."
echo ""

# CPU Top 20
echo "=== CPU Top 20 (按 flat 时间排序) ===" > "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top -cum "$OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Heap Top 20
echo "=== Heap Top 20 (按分配大小排序) ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Goroutine 统计
echo "=== Goroutine 统计 ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/goroutine_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Mutex 统计
echo "=== Mutex 等待时间 ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/mutex_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Block 统计
echo "=== Block 等待时间 ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/block_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true

echo "分析报告已保存到: $OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
echo ""

# 显示关键信息
echo "=========================================="
echo "关键性能指标摘要"
echo "=========================================="
echo ""
echo "查看详细报告:"
echo "  cat $OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
echo ""
echo "交互式查看 CPU Profile:"
echo "  go tool pprof $OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz"
echo ""
echo "交互式查看 Heap Profile:"
echo "  go tool pprof $OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz"
echo ""
echo "生成火焰图 (需要安装 go-torch 或 pprof):"
echo "  go tool pprof -http=:8080 $OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz"
echo ""

# 如果启动了服务器，询问是否关闭
if [ -n "$SERVER_PID" ]; then
    read -p "是否关闭服务器？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $SERVER_PID 2>/dev/null || true
        echo "服务器已关闭"
    fi
fi


