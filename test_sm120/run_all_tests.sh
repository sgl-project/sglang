#!/bin/bash
# Full E2E test suite for SM120 DSv4-Flash (rebased on main)
# Usage: bash run_all_tests.sh
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
VENV=/home/scratch.alichen_sw_1/workspace/sglang_venv
PY=$VENV/bin/python3
export HF_HOME=/home/scratch.alichen_sw_1/.hf_cache
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_HACK_FLASHMLA_BACKEND=kernel
export PYTHONUNBUFFERED=1
export CUDA_HOME=/home/scratch.alichen_sw_1/workspace/cuda-12.8
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA 13 libnvrtc needed by sgl_kernel
export LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO_DIR/python:$PYTHONPATH

SERVER_PORT=30000
SERVER_URL="http://localhost:${SERVER_PORT}"

echo "=================================================="
echo " SM120 DSv4-Flash Rebase Validation"
echo " $(date)"
echo "=================================================="

# Step 1: Install rebased sglang
echo ""
echo "[1/4] Installing rebased sglang..."
cd $REPO_DIR
pip install -e "python[all]" --no-build-isolation 2>&1 | tail -3

# Step 2: Start server
echo ""
echo "[2/4] Starting sglang server (TP=8, CUDA graph enabled)..."
$PY -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --tp 8 \
    --trust-remote-code \
    --mem-fraction-static 0.70 \
    --port $SERVER_PORT \
    --host 0.0.0.0 \
    --cuda-graph-max-bs 32 \
    --watchdog-timeout 600 \
    > $SCRIPT_DIR/server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s $SERVER_URL/health > /dev/null 2>&1; then
        echo "  Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  ERROR: Server process died. Check server.log"
        tail -50 $SCRIPT_DIR/server.log
        exit 1
    fi
    sleep 5
done

if ! curl -s $SERVER_URL/health > /dev/null 2>&1; then
    echo "  ERROR: Server not ready after 600s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Step 3: E2E Benchmark
echo ""
echo "[3/4] Running E2E benchmark (ISL~4K, OSL=8, BS=1,4,8,16,32)..."
cd $SCRIPT_DIR
$PY bench_e2e.py \
    --max-tokens 8 \
    --warmup 2 \
    --iters 5 \
    --batch-sizes "1,4,8,16,32" \
    --output bench_e2e_isl4k_osl8.json \
    2>&1 | tee bench_e2e.log

# Step 4: GSM8K correctness
echo ""
echo "[4/4] Running GSM8K correctness validation (200 questions)..."
$PY eval_gsm8k.py \
    --max-questions 200 \
    --max-tokens 512 \
    --workers 4 \
    --tp 8 \
    2>&1 | tee gsm8k_eval.log

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "=================================================="
echo " All tests complete! Results:"
echo "  - bench_e2e_isl4k_osl8.json"
echo "  - gsm8k_full_tp8.json"
echo "=================================================="
