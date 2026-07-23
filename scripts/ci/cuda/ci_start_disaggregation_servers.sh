#!/bin/bash
set -euo pipefail

# Optional: set DISAGG_READY_FILE to a filepath; when all servers are healthy, the script will
# create this file as a readiness signal (useful for CI to proceed to next steps).
DISAGG_READY_FILE="${DISAGG_READY_FILE:-}"

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

# Launch decode servers on GPU 4–7
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

# Wait for disaggregation servers to initialize
echo "Waiting for disaggregation servers to initialize..."

# Health check with 5-minute timeout
TIMEOUT=300
START_TIME=$(date +%s)

echo "Checking health of all 8 servers..."
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ Timeout: Servers did not become healthy within 5 minutes"
        exit 1
    fi

    HEALTHY_COUNT=0
    # Check all 8 servers (127.0.0.1-8:30001-30008)
    for i in {1..8}; do
        if curl -s -f "http://127.0.0.$i:$((30000 + i))/health" >/dev/null 2>&1; then
            HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
        fi
    done

    echo "Healthy servers: $HEALTHY_COUNT/8 (elapsed: ${ELAPSED}s)"

    if [ $HEALTHY_COUNT -eq 8 ]; then
        echo "✅ All 8 servers are healthy!"
        # Emit readiness signal file if requested
        if [ -n "$DISAGG_READY_FILE" ]; then
            echo "Creating readiness flag: $DISAGG_READY_FILE"
            # Ensure parent dir exists; ignore errors
            mkdir -p "$(dirname "$DISAGG_READY_FILE")" 2>/dev/null || true
            touch "$DISAGG_READY_FILE"
        fi
        break
    else
        sleep 10  # Wait 10 seconds before next check
    fi
done

# Don't launch router here - just keep servers running
echo "✅ All disaggregation servers are ready and waiting for router connections"

# Keep the script running
wait
