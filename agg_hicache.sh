#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Default values
MODEL="Qwen/Qwen3-0.6B"
ENABLE_OTEL=false

# the model we use is small so we set small mem frac and then set hicache size to 3 (3gb)

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 64  \
  --tp 1 \
  --mem-fraction-static 0.03 \
  --trust-remote-code \
  --enable-metrics \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --enable-hierarchical-cache \
  --hicache-size 3 --enable-cache-report --hicache-storage-backend file
