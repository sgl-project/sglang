#!/bin/bash

# 简单的性能测试脚本，用于在收集 pprof 数据时发送请求

set -e

SERVER_URL=${SERVER_URL:-"http://localhost:8080"}
DURATION=${DURATION:-30}  # 测试持续时间（秒）
CONCURRENT=${CONCURRENT:-1}  # 并发请求数

echo "=========================================="
echo "性能测试脚本"
echo "=========================================="
echo "SERVER_URL: $SERVER_URL"
echo "DURATION: ${DURATION}s"
echo "CONCURRENT: $CONCURRENT"
echo ""

# 测试请求 JSON
TEST_REQUEST='{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": true,
  "max_tokens": 100
}'

# 检查服务器是否可用
if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "错误: 服务器不可用 (${SERVER_URL}/health)"
    exit 1
fi

echo "开始发送测试请求..."
echo ""

# 发送流式请求的函数
send_stream_request() {
    local request_num=$1
    local start_time=$(date +%s.%N)
    
    curl -s -N -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$TEST_REQUEST" \
        > /dev/null 2>&1
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    echo "请求 $request_num 完成，耗时: ${duration}s"
}

# 并发发送请求
if [ "$CONCURRENT" -eq 1 ]; then
    # 单线程模式：持续发送请求
    end_time=$(($(date +%s) + DURATION))
    request_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        request_count=$((request_count + 1))
        send_stream_request $request_count
    done
    
    echo ""
    echo "测试完成，共发送 $request_count 个请求"
else
    # 多线程模式：并发发送请求
    end_time=$(($(date +%s) + DURATION))
    request_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        # 启动并发请求
        for i in $(seq 1 $CONCURRENT); do
            request_count=$((request_count + 1))
            send_stream_request $request_count &
        done
        
        # 等待所有请求完成
        wait
        
        # 短暂休息，避免过载
        sleep 0.1
    done
    
    echo ""
    echo "测试完成，共发送 $request_count 个请求"
fi


