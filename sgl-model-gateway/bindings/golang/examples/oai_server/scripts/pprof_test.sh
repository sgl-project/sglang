#!/bin/bash

# Simple performance test script for sending requests while collecting pprof data

set -e

SERVER_URL=${SERVER_URL:-"http://localhost:8080"}
DURATION=${DURATION:-30}  # Test duration (seconds)
CONCURRENT=${CONCURRENT:-1}  # Number of concurrent requests

echo "=========================================="
echo "Performance Test Script"
echo "=========================================="
echo "SERVER_URL: $SERVER_URL"
echo "DURATION: ${DURATION}s"
echo "CONCURRENT: $CONCURRENT"
echo ""

# Test request JSON
TEST_REQUEST='{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": true,
  "max_tokens": 100
}'

# Check if server is available
if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Error: Server not available (${SERVER_URL}/health)"
    exit 1
fi

echo "Starting to send test requests..."
echo ""

# Function to send streaming request
send_stream_request() {
    local request_num=$1
    local start_time=$(date +%s.%N)

    curl -s -N -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$TEST_REQUEST" \
        > /dev/null 2>&1

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    echo "Request $request_num completed, duration: ${duration}s"
}

# Send requests concurrently
if [ "$CONCURRENT" -eq 1 ]; then
    # Single-threaded mode: continuously send requests
    end_time=$(($(date +%s) + DURATION))
    request_count=0

    while [ $(date +%s) -lt $end_time ]; do
        request_count=$((request_count + 1))
        send_stream_request $request_count
    done

    echo ""
    echo "Test completed, sent $request_count requests"
else
    # Multi-threaded mode: send requests concurrently
    end_time=$(($(date +%s) + DURATION))
    request_count=0

    while [ $(date +%s) -lt $end_time ]; do
        # Start concurrent requests
        for i in $(seq 1 $CONCURRENT); do
            request_count=$((request_count + 1))
            send_stream_request $request_count &
        done

        # Wait for all requests to complete
        wait

        # Brief rest to avoid overload
        sleep 0.1
    done

    echo ""
    echo "Test completed, sent $request_count requests"
fi
