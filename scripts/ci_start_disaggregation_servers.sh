#!/bin/bash

MODEL_PATH="/raid/models/meta-llama/Llama-3.1-8B-Instruct"

# Function to find the first available active IB device
find_active_ib_device() {
    ibv_devices=$(ibv_devices)
    for device in $ibv_devices; do
        if ibv_devinfo $device >/dev/null 2>&1; then
            state=$(ibv_devinfo $device | grep "state:" | head -1 | awk '{print $2}')
            if [[ "$state" == "PORT_ACTIVE" ]]; then
                echo "$device"
                return 0
            fi
        fi
    done
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
for i in {0..7}; do
  HOST="127.0.0.$((i + 1))"
  PORT=$((30001 + i))
  until curl --fail --silent $HOST:$PORT/health; do
    if ((SECONDS > 120)); then
      echo "Timeout waiting for $HOST:$PORT to become healthy"
      exit 1
    fi
    echo "Waiting for $HOST:$PORT to become healthy..."
    sleep 5
  done
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

wait
