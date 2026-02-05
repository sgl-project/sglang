#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HiCache with Mooncake storage backend
# Starts mooncake_master + SGLang server, kills all on Ctrl+C

set -e

# Track all PIDs
PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Also kill any child processes
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete."
    exit 0
}

trap cleanup EXIT INT TERM

# Configuration
MODEL="Qwen/Qwen3-0.6B"
MOONCAKE_MASTER_PORT=50051
MOONCAKE_METADATA_PORT=8080

echo "=============================================="
echo "HiCache + Mooncake Storage Backend"
echo "=============================================="

# Step 1: Start Mooncake Master with embedded metadata server
echo ""
echo "[1/2] Starting Mooncake Master..."
mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=$MOONCAKE_METADATA_PORT \
    --eviction_high_watermark_ratio=0.95 &
PIDS+=($!)
echo "  Mooncake Master PID: ${PIDS[-1]}"

# Wait for mooncake to be ready
sleep 3

# Step 2: Start SGLang server with mooncake backend
echo ""
echo "[2/2] Starting SGLang Server with Mooncake backend..."

# Mooncake config as JSON
MOONCAKE_CONFIG=$(cat <<EOF
{
    "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
    "protocol": "tcp",
    "device_name": "",
    "global_segment_size": 3221225472
}
EOF
)

python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --page-size 64 \
    --tp 1 \
    --mem-fraction-static 0.03 \
    --trust-remote-code \
    --enable-metrics \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
    --enable-hierarchical-cache \
    --hicache-size 3 \
    --enable-cache-report \
    --hicache-storage-backend mooncake \
    --hicache-storage-backend-extra-config "$MOONCAKE_CONFIG" &
PIDS+=($!)
echo "  SGLang Server PID: ${PIDS[-1]}"

echo ""
echo "=============================================="
echo "All services started. PIDs: ${PIDS[*]}"
echo "Press Ctrl+C to stop all services."
echo "=============================================="

# Wait for any process to exit
wait -n "${PIDS[@]}" 2>/dev/null || true

echo "A process exited unexpectedly."
cleanup
