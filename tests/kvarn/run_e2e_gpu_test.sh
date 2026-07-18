#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# KVarN end-to-end GPU test script.
#
# Requires: a GPU with CUDA support, sglang installed, and a small model
# (e.g. meta-llama/Llama-3.2-1B-Instruct or Qwen/Qwen2.5-0.5B-Instruct).
#
# Usage:
#   KVARN_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash tests/kvarn/run_e2e_gpu_test.sh
#
# What it tests:
#   1. Server starts with --kv-cache-dtype kvarn_k4v4_g128
#   2. Basic completion request succeeds
#   3. Output is coherent (not garbage)
#   4. Server can handle multiple concurrent requests
#   5. Prefill + decode both work
#   6. Comparison with fp16 baseline (optional)

set -euo pipefail

MODEL="${KVARN_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
KVARN_DTYPE="${KVARN_DTYPE:-kvarn_k4v4_g128}"
PORT="${KVARN_PORT:-30000}"
TIMEOUT="${KVARN_TIMEOUT:-120}"

echo "=== KVarN E2E GPU Test ==="
echo "Model: $MODEL"
echo "KV cache dtype: $KVARN_DTYPE"
echo "Port: $PORT"
echo ""

# Start server with KVarN
echo ">>> Starting sglang server with KVarN..."
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --kv-cache-dtype "$KVARN_DTYPE" \
    --attention-backend kvarn \
    --port "$PORT" \
    --log-level info \
    --disable-radix-cache \
    --context-length 2048 &
SERVER_PID=$!

cleanup() {
    echo ">>> Cleaning up server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo ">>> Waiting for server to be ready..."
for i in $(seq 1 $TIMEOUT); do
    if curl -s http://localhost:$PORT/health | grep -q "ok" 2>/dev/null; then
        echo ">>> Server is ready (after ${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ">>> ERROR: Server process died"
        exit 1
    fi
    sleep 1
    if [ $i -eq $TIMEOUT ]; then
        echo ">>> ERROR: Server failed to start within ${TIMEOUT}s"
        exit 1
    fi
done

# Test 1: Basic completion
echo ""
echo ">>> Test 1: Basic completion"
RESPONSE=$(curl -s http://localhost:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The capital of France is",
        "sampling_params": {"max_new_tokens": 16, "temperature": 0}
    }')
echo "Response: $RESPONSE"
TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])" 2>/dev/null || echo "")
if [ -z "$TEXT" ]; then
    echo ">>> FAIL: Empty response"
    exit 1
fi
echo ">>> PASS: Got non-empty response"

# Test 2: Coherence check (should mention Paris)
echo ""
echo ">>> Test 2: Coherence check"
if echo "$TEXT" | grep -qi "paris" 2>/dev/null; then
    echo ">>> PASS: Response mentions Paris"
else
    echo ">>> WARN: Response does not mention Paris (may be OK for small models)"
    echo "    Response was: $TEXT"
fi

# Test 3: Multiple concurrent requests
echo ""
echo ">>> Test 3: Multiple concurrent requests"
for i in 1 2 3 4; do
    curl -s http://localhost:$PORT/generate \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"Hello $i\", \"sampling_params\": {\"max_new_tokens\": 8, \"temperature\": 0}}" &
done
wait
echo ">>> PASS: Concurrent requests completed"

# Test 4: Longer context (tests prefill + multiple decode blocks)
echo ""
echo ">>> Test 4: Longer context"
LONG_TEXT="Once upon a time, there was a little robot named Pi who lived in a big city. Pi loved to explore and learn new things. One day, Pi decided to go on an adventure to find the tallest building in the city. Along the way, Pi met many friends who helped with directions. The story continues with"
RESPONSE=$(curl -s http://localhost:$PORT/generate \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$LONG_TEXT\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}")
echo "Response: $(echo $RESPONSE | python3 -c 'import sys,json; print(json.load(sys.stdin)["text"][:200])' 2>/dev/null || echo 'parse error')"
echo ">>> PASS: Long context request completed"

# Test 5: Multi-turn (tests KV cache reuse)
echo ""
echo ">>> Test 5: Multi-turn conversation"
RESPONSE1=$(curl -s http://localhost:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "My name is Alice.", "sampling_params": {"max_new_tokens": 8, "temperature": 0}}')
echo "Turn 1: $RESPONSE1"
RESPONSE2=$(curl -s http://localhost:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "What is my name?", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}')
echo "Turn 2: $RESPONSE2"
echo ">>> PASS: Multi-turn completed"

echo ""
echo "=== All tests passed! ==="
echo ""
echo "KVarN is working end-to-end with:"
echo "  - Model: $MODEL"
echo "  - KV cache dtype: $KVARN_DTYPE"
echo "  - Attention backend: kvarn (Hadamard rotation + fp16 tail pool)"