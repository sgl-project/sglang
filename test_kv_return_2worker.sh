#!/bin/bash
# 2-Worker Disaggregated KV Return Test
# Runs prefill + decode workers on a single GPU with NIXL TCP transport
# Validates: forward KV transfer (P→D), reverse KV return (D→P)

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
echo "  All components ready — sending test request"
echo "=========================================="
echo ""

# Send a chat completion through the router/LB
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}\n" \
    http://localhost:$LB_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 32,
        "temperature": 0
    }')

echo "Response:"
echo "$RESPONSE"
echo ""

# Give the async KV return a moment to complete
sleep 3

echo "=========================================="
echo "  KV Return Evidence Check"
echo "=========================================="
echo ""

echo "--- [PREFILL] enable_kv_return in ServerArgs ---"
grep -i "enable_kv_return" "$LOG_DIR/prefill.log" | head -5 || echo "(not found)"
echo ""

echo "--- [PREFILL] Page pre-allocation ---"
grep -i "Pre-allocated.*pages.*KV return" "$LOG_DIR/prefill.log" | head -5 || echo "(not found)"
echo ""

echo "--- [PREFILL] KV return receiver started ---"
grep -i "KV return.*prefill\|prefill.*accept.*returned" "$LOG_DIR/prefill.log" | head -5 || echo "(not found)"
echo ""

echo "--- [PREFILL] Sent KV return info to decode ---"
grep -i "Sent KV return registration" "$LOG_DIR/prefill.log" | head -5 || echo "(not found)"
echo ""

echo "--- [DECODE] KV return listener started ---"
grep -i "KV return listener started\|KV return enabled.*decode" "$LOG_DIR/decode.log" | head -5 || echo "(not found)"
echo ""

echo "--- [DECODE] Registered prefill for KV return ---"
grep -i "Registered prefill agent" "$LOG_DIR/decode.log" | head -5 || echo "(not found)"
echo ""

echo "--- [DECODE] KV return transfers ---"
grep -i "KV return" "$LOG_DIR/decode.log" | head -10 || echo "(none)"
echo ""

echo "--- [PREFILL] KV return notifications received ---"
grep -i "KV return" "$LOG_DIR/prefill.log" | head -10 || echo "(none)"
echo ""

echo "--- [ROUTER] Routing activity ---"
tail -20 "$LOG_DIR/router.log" || echo "(empty)"
echo ""

echo "=========================================="
echo "  Test Complete"
echo "=========================================="
