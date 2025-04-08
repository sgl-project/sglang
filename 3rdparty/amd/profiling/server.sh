#!/bin/bash

# export SGLANG_TORCH_PROFILER_DIR=/data/sglang/
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/profile/

# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define the log file with a timestamp
LOGFILE="sglang_server_log_$TIMESTAMP.json"

# Run the Python command and save the output to the log file
loadTracer.sh python3 -m sglang.launch_server \
    --model-path /sgl-workspace/sglang/dummy_grok1 \
    --tokenizer-path Xenova/grok-1-tokenizer \
    --load-format dummy \
    --quantization fp8 \
    --tp 8 \
    --port 30000 \
    --disable-radix-cache 2>&1 | tee "$LOGFILE"
