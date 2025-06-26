#!/bin/bash

MODEL_PATH="/raid/models/meta-llama/Llama-3.1-8B-Instruct"

# Function to find the first available active IB device
find_active_ib_device() {
    for device in mlx5_{0..11}; do
        if ibv_devinfo $device >/dev/null 2>&1; then
            state=$(ibv_devinfo $device | grep "state:" | head -1 | awk '{print $2}')
            if [[ "$state" == "PORT_ACTIVE" ]]; then
                echo "$device"
                return 0
            fi
        fi
    done
    echo "No active IB device found" >&2
    return 1
}

# Get the first available active IB device
DEVICE=$(find_active_ib_device)
echo "Using IB device: $DEVICE"

# Launch prefill servers on GPU 0–3
echo "=== STEP 1: Launching prefill servers ==="
for i in {0..3}; do
  PORT=$((30001 + i))
  BOOTSTRAP_PORT=$((9001 + i))
  HOST="127.0.0.$((i + 1))"
  echo "Launching PREFILL server on GPU $i at $HOST:$PORT (bootstrap: $BOOTSTRAP_PORT)"
  CUDA_VISIBLE_DEVICES=$i \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode prefill \
    --host "$HOST" \
    --port "$PORT" \
    --disaggregation-ib-device "$DEVICE" \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" &
done

# Wait for prefill servers to be ready
echo "=== STEP 2: Waiting for prefill servers to be ready ==="
TIMEOUT=300
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ Timeout: Prefill servers did not become ready within 5 minutes"
        exit 1
    fi

    READY_COUNT=0
    # Check prefill servers (127.0.0.1-4:30001-30004)
    for i in {1..4}; do
        if curl --connect-timeout 5 --silent "http://127.0.0.$i:$((30000 + i))" >/dev/null 2>&1; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done

    echo "Ready prefill servers: $READY_COUNT/4 (elapsed: ${ELAPSED}s)"

    if [ $READY_COUNT -eq 4 ]; then
        echo "✅ All 4 prefill servers are ready!"
        break
    else
        sleep 10
    fi
done

# Launch decode servers on GPU 4–7
echo "=== STEP 3: Launching decode servers ==="
for i in {4..7}; do
  PORT=$((30001 + i))
  HOST="127.0.0.$((i + 1))"
  echo "Launching DECODE server on GPU $i at $HOST:$PORT"
  CUDA_VISIBLE_DEVICES=$i \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode decode \
    --host "$HOST" \
    --port "$PORT" \
    --disaggregation-ib-device "$DEVICE" \
    --base-gpu-id 0 &
done

# Wait for decode servers to be ready
echo "=== STEP 4: Waiting for decode servers to be ready ==="
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ Timeout: Decode servers did not become ready within 5 minutes"
        exit 1
    fi

    READY_COUNT=0
    # Check decode servers (127.0.0.5-8:30005-30008)
    for i in {5..8}; do
        if curl --connect-timeout 5 --silent "http://127.0.0.$i:$((30000 + i))" >/dev/null 2>&1; then
            READY_COUNT=$((READY_COUNT + 1))
        fi
    done

    echo "Ready decode servers: $READY_COUNT/4 (elapsed: ${ELAPSED}s)"

    if [ $READY_COUNT -eq 4 ]; then
        echo "✅ All 4 decode servers are ready!"
        break
    else
        sleep 10
    fi
done

# Launch the router
echo "Launching router at 127.0.0.9:8000..."
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy power_of_two \
  --prefill http://127.0.0.1:30001 9001 \
  --prefill http://127.0.0.2:30002 9002 \
  --prefill http://127.0.0.3:30003 9003 \
  --prefill http://127.0.0.4:30004 9004 \
  --decode http://127.0.0.5:30005 \
  --decode http://127.0.0.6:30006 \
  --decode http://127.0.0.7:30007 \
  --decode http://127.0.0.8:30008 \
  --host 127.0.0.9 \
  --port 8000 &

wait  # Wait for all background jobs to finish
