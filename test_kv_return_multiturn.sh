#!/bin/bash
# Multi-Turn Disaggregated KV Return Test
# Validates that KV returned from decode→prefill gets inserted into RadixCache
# by checking cached_tokens across sequential identical requests.
#
# Flow:
#   1. Send prompt A (cold — no cache hit expected)
#   2. Wait for async KV return transfer
#   3. Send prompt A again (warm — cached_tokens > 0 if KV return works)
#   4. Send prompt B with shared prefix (partial cache hit possible)

set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
HF_CACHE="/scratch/hf_cache"
LOG_DIR="/scratch/kv_return_test"
PREFILL_PORT=30000
DECODE_PORT=30001
LB_PORT=30100
# Low memory fraction since both workers share one GPU
MEM_FRAC=0.08

mkdir -p "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"

PREFILL_PID="" DECODE_PID="" LB_PID="" ETCD_PID="" NATS_PID=""

cleanup() {
    echo "=== Shutting down ==="
    kill $PREFILL_PID $DECODE_PID $LB_PID $ETCD_PID $NATS_PID 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done. Full logs in $LOG_DIR/"
}
trap cleanup EXIT

echo "=== Starting etcd ==="
etcd --data-dir /tmp/etcd-data --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://127.0.0.1:2379 > "$LOG_DIR/etcd.log" 2>&1 &
ETCD_PID=$!
sleep 2

echo "=== Starting nats ==="
nats-server -js -sd /tmp/nats-data > "$LOG_DIR/nats.log" 2>&1 &
NATS_PID=$!
sleep 2

echo "=== Starting prefill worker on port $PREFILL_PORT ==="
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PREFILL_PORT" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --enable-kv-return \
    --kv-return-budget-fraction 0.1 \
    --enable-cache-report \
    --trust-remote-code \
    --log-level info \
    --mem-fraction-static "$MEM_FRAC" \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "Prefill PID: $PREFILL_PID"

# Stagger startup to let prefill finish GPU init before decode claims memory
sleep 30

echo "=== Starting decode worker on port $DECODE_PORT ==="
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$DECODE_PORT" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --enable-kv-return \
    --trust-remote-code \
    --log-level info \
    --mem-fraction-static "$MEM_FRAC" \
    > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "Decode PID: $DECODE_PID"

echo "=== Waiting for servers to be ready (up to 120s each) ==="

# Wait for prefill
for attempt in $(seq 1 120); do
    if curl -s http://localhost:$PREFILL_PORT/health > /dev/null 2>&1; then
        echo "Prefill ready after ${attempt}s"
        break
    fi
    if ! kill -0 $PREFILL_PID 2>/dev/null; then
        echo "ERROR: Prefill process died. Last 50 lines:"
        tail -50 "$LOG_DIR/prefill.log"
        exit 1
    fi
    sleep 1
done

# Wait for decode
for attempt in $(seq 1 120); do
    if curl -s http://localhost:$DECODE_PORT/health > /dev/null 2>&1; then
        echo "Decode ready after ${attempt}s"
        break
    fi
    if ! kill -0 $DECODE_PID 2>/dev/null; then
        echo "ERROR: Decode process died. Last 50 lines:"
        tail -50 "$LOG_DIR/decode.log"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Starting router/load balancer on port $LB_PORT ==="
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --prefill "http://localhost:$PREFILL_PORT" \
    --decode "http://localhost:$DECODE_PORT" \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &
LB_PID=$!
echo "Router PID: $LB_PID"

# Wait for router to be ready
for attempt in $(seq 1 30); do
    if curl -s http://localhost:$LB_PORT/health > /dev/null 2>&1; then
        echo "Router ready after ${attempt}s"
        break
    fi
    if ! kill -0 $LB_PID 2>/dev/null; then
        echo "ERROR: Router process died. Last 30 lines:"
        tail -30 "$LOG_DIR/router.log"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=========================================="
echo "  All components ready — sending test requests"
echo "=========================================="
echo ""

# Helper: extract a JSON field using python3 (no jq in container)
json_get() {
    python3 -c "
import json, sys
data = json.load(sys.stdin)
keys = '$1'.split('.')
v = data
for k in keys:
    if isinstance(v, dict) and k in v:
        v = v[k]
    elif isinstance(v, list) and k.isdigit() and int(k) < len(v):
        v = v[int(k)]
    else:
        v = '$2'
        break
print(v)
"
}

# Helper: send a chat completion and extract metrics
REQ1_CACHED=0 REQ1_PROMPT=0
REQ2_CACHED=0 REQ2_PROMPT=0
REQ3_CACHED=0 REQ3_PROMPT=0

send_request() {
    local prompt="$1"
    local label="$2"
    echo "--- $label ---"
    echo "Prompt: $prompt"

    local response
    response=$(curl -s http://localhost:$LB_PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "'"$prompt"'"}],
            "max_tokens": 32,
            "temperature": 0
        }')

    # Check for valid response
    local has_id
    has_id=$(echo "$response" | json_get "id" "")
    if [ -z "$has_id" ]; then
        echo "ERROR: Bad response: $response"
        return 1
    fi

    local cached_tokens prompt_tokens content
    cached_tokens=$(echo "$response" | json_get "usage.prompt_tokens_details.cached_tokens" "0")
    prompt_tokens=$(echo "$response" | json_get "usage.prompt_tokens" "?")
    content=$(echo "$response" | json_get "choices.0.message.content" "(empty)")

    echo "  prompt_tokens:  $prompt_tokens"
    echo "  cached_tokens:  $cached_tokens"
    echo "  response:       $content"
    echo ""

    # Store in global vars for later validation
    eval "${label}_CACHED=$cached_tokens"
    eval "${label}_PROMPT=$prompt_tokens"
}

# ─── Request 1: Cold start (no cache expected) ───
send_request "What is the capital of France? Answer in one word." "REQ1"

# Wait for async KV return transfer to complete (decode→prefill)
echo "Waiting 5s for KV return transfer..."
sleep 5

# ─── Request 2: Same prompt (should hit cache if KV return + RadixCache works) ───
send_request "What is the capital of France? Answer in one word." "REQ2"

# Wait briefly for any additional transfer
sleep 3

# ─── Request 3: Different prompt with shared prefix (partial cache hit possible) ───
send_request "What is the capital of Germany? Answer in one word." "REQ3"

echo ""
echo "=========================================="
echo "  KV Return Log Evidence"
echo "=========================================="
echo ""

echo "--- [DECODE] KV return transfers ---"
grep -i "KV return completed" "$LOG_DIR/decode.log" | tail -10 || echo "(none)"
echo ""

echo "--- [PREFILL] KV return metadata received ---"
grep -i "KV return metadata" "$LOG_DIR/prefill.log" | tail -10 || echo "(none)"
echo ""

echo "--- [PREFILL] RadixCache insertions ---"
grep -i "Processed.*KV return insertions\|tree already has full sequence" "$LOG_DIR/prefill.log" | tail -10 || echo "(none)"
echo ""

echo "--- [PREFILL] Memory accounting ---"
grep -i "memory_leak\|kv_return_reserved" "$LOG_DIR/prefill.log" | tail -5 || echo "(clean — no memory leak)"
echo ""

echo "=========================================="
echo "  Validation (log-based)"
echo "=========================================="
echo ""

# In disagg mode, cached_tokens in the API response comes from the decode
# worker which has no visibility into prefill-side cache state. So we validate
# by checking server logs instead.
PASS=true
FAIL_REASONS=""

# Check 1: Decode worker sent KV returns
DECODE_RETURNS=$({ grep -c "KV return completed" "$LOG_DIR/decode.log" 2>/dev/null || true; })
echo "  [CHECK 1] Decode KV return transfers: $DECODE_RETURNS"
if [ "$DECODE_RETURNS" -ge 1 ]; then
    echo "    PASS — decode sent KV back to prefill"
else
    echo "    FAIL — no KV return transfers from decode"
    PASS=false
    FAIL_REASONS="$FAIL_REASONS\n  - No KV return transfers from decode"
fi
echo ""

# Check 2: Prefill inserted into RadixCache
PREFILL_INSERTIONS=$({ grep -c "Processed.*KV return insertions" "$LOG_DIR/prefill.log" 2>/dev/null || true; })
echo "  [CHECK 2] Prefill RadixCache insertions: $PREFILL_INSERTIONS"
if [ "$PREFILL_INSERTIONS" -ge 1 ]; then
    echo "    PASS — prefill inserted returned KV into RadixCache"
else
    echo "    FAIL — no RadixCache insertions on prefill"
    PASS=false
    FAIL_REASONS="$FAIL_REASONS\n  - No RadixCache insertions on prefill"
fi
echo ""

# Check 3: No memory leak crash (prefill survived all requests)
MEMORY_LEAKS=$({ grep -c "memory leak detected" "$LOG_DIR/prefill.log" 2>/dev/null || true; })
echo "  [CHECK 3] Memory leak errors: $MEMORY_LEAKS"
if [ "$MEMORY_LEAKS" -eq 0 ]; then
    echo "    PASS — no memory accounting errors"
else
    echo "    FAIL — memory leak detection triggered"
    PASS=false
    FAIL_REASONS="$FAIL_REASONS\n  - Memory leak detection errors in prefill"
fi
echo ""

# Check 4: All 3 requests got valid responses
echo "  [CHECK 4] API responses:"
echo "    REQ1: prompt_tokens=$REQ1_PROMPT (expected: >0)"
echo "    REQ2: prompt_tokens=$REQ2_PROMPT (expected: >0)"
echo "    REQ3: prompt_tokens=$REQ3_PROMPT (expected: >0)"
ALL_RESPONDED=true
for req in REQ1 REQ2 REQ3; do
    val=$(eval echo "\$${req}_PROMPT")
    if [ -z "$val" ] || [ "$val" = "?" ] || [ "$val" = "0" ]; then
        ALL_RESPONDED=false
    fi
done
if $ALL_RESPONDED; then
    echo "    PASS — all requests completed successfully"
else
    echo "    FAIL — some requests did not complete"
    PASS=false
    FAIL_REASONS="$FAIL_REASONS\n  - Not all requests returned valid responses"
fi
echo ""

if $PASS; then
    echo "=========================================="
    echo "  PASS: KV return pipeline working!"
    echo "  - $DECODE_RETURNS decode→prefill transfers"
    echo "  - $PREFILL_INSERTIONS RadixCache insertions"
    echo "  - 0 memory errors"
    echo "  - All requests completed"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "  FAIL: KV return validation failed"
    echo -e "  Reasons:$FAIL_REASONS"
    echo "=========================================="
    exit 1
fi
