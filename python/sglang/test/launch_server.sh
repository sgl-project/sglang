#!/bin/bash

# =================================================================
# SGLang Server Launch Script for Qwen2.5-VL with Pipeline Parallelism
# =================================================================
#
# Instructions:
# 1. IMPORTANT: Update the `MODEL_PATH` variable below to point to the
#    correct location of your Qwen2.5-VL-32B-Instruct model directory.
#
# 2. Make this script executable:
#    chmod +x launch_server.sh
#
# 3. Run the script:
#    ./launch_server.sh
#
# You can experiment by changing the --pp-size parameter (e.g., to 1, 2, 4).
# =================================================================

# --- Configuration ---
# !!! PLEASE UPDATE THIS PATH !!!
MODEL_PATH="/path/to/your/Qwen2.5-VL-32B-Instruct/"

# --- Launch Command ---
echo "================================================================"
echo "Starting SGLang server for model: $MODEL_PATH"
echo "Pipeline Parallelism (PP) size: 2"
echo "================================================================"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port 26000 \
    --max-running-requests 4 \
    --mem-fraction-static 0.95 \
    --quantization fp8 \
    --tp-size 1 \
    --pp-size 2 \
    --disable-radix-cache \
    --disable-overlap-schedule 