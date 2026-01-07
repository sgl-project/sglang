#!/bin/bash
# Torch Profile Benchmark Script for Shared Expert Load Balancing
#
# Captures torch profiles for each experiment configuration
# Uses reduced num_prompts for faster profiling

set -e

MODEL_PATH="/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/"
HOST="0.0.0.0"
PORT=30000
RESULT_DIR="/lustre/raplab/client/xutingz/workspace/bench/torch_profile/$(date +%Y%m%d_%H%M%S)"
PROFILE_DIR="${RESULT_DIR}/profiles"

# Benchmark parameters (same as log collection, but fewer prompts for profiling)
NUM_PROMPTS=128
RANDOM_INPUT=1024
RANDOM_OUTPUT=1
MAX_CONCURRENCY=32

mkdir -p ${RESULT_DIR}
mkdir -p ${PROFILE_DIR}

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -q 'DeepSeek-V3'; then
            echo "Server is ready!"
            return 0
        fi
        echo "  Still waiting... ($i/60)"
        sleep 10
    done
    echo "Server failed to start!"
    return 1
}

kill_server() {
    echo "Stopping server..."
    pkill -f "launch_server" 2>/dev/null || true
    sleep 5
}

run_benchmark_with_profile() {
    local name=$1
    local output_file="${RESULT_DIR}/${name}.jsonl"
    local exp_profile_dir="${PROFILE_DIR}/${name}"

    mkdir -p ${exp_profile_dir}

    echo "Running benchmark with torch profile: ${name}"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --num-prompts ${NUM_PROMPTS} \
        --random-input ${RANDOM_INPUT} \
        --random-output ${RANDOM_OUTPUT} \
        --max-concurrency ${MAX_CONCURRENCY} \
        --model ${MODEL_PATH} \
        --output-file ${output_file} \
        --profile \
        --profile-num-steps 10

    # Move profile files to experiment directory
    mv ${PROFILE_DIR}/*.json ${exp_profile_dir}/ 2>/dev/null || true
    mv /tmp/sglang_torch_profiler*/*.json ${exp_profile_dir}/ 2>/dev/null || true

    echo "Results saved to: ${output_file}"
    echo "Profile saved to: ${exp_profile_dir}/"
}

extract_metrics() {
    local file=$1
    python3 -c "
import json
with open('${file}') as f:
    d = json.load(f)
print(f\"  Output Throughput: {d['output_throughput']:.2f} tok/s\")
print(f\"  Mean E2E Latency: {d['mean_e2e_latency_ms']:.0f} ms\")
print(f\"  Mean TPOT: {d['mean_tpot_ms']:.2f} ms\")
print(f\"  Mean TTFT: {d['mean_ttft_ms']:.2f} ms\")
"
}

echo "=========================================="
echo "Torch Profile Benchmark"
echo "=========================================="
echo "Parameters:"
echo "  NUM_PROMPTS: ${NUM_PROMPTS}"
echo "  RANDOM_INPUT: ${RANDOM_INPUT}"
echo "  RANDOM_OUTPUT: ${RANDOM_OUTPUT}"
echo "  MAX_CONCURRENCY: ${MAX_CONCURRENCY}"
echo "  RESULT_DIR: ${RESULT_DIR}"
echo ""

# ==========================================
# Experiment 1: Shared Expert TP8 (baseline)
# ==========================================
echo "=========================================="
echo "Experiment 1: Shared Expert TP8 (Baseline)"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp1_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp1_tp8_baseline"

echo ""
echo "Experiment 1 Results:"
extract_metrics "${RESULT_DIR}/exp1_tp8_baseline.jsonl"
echo ""

# ==========================================
# Experiment 2: Shared Expert DP + Uniform
# ==========================================
echo "=========================================="
echo "Experiment 2: Shared Expert DP + Uniform"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode uniform \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp2_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp2_dp_uniform"

echo ""
echo "Experiment 2 Results:"
extract_metrics "${RESULT_DIR}/exp2_dp_uniform.jsonl"
echo ""

# ==========================================
# Experiment 3: Shared Expert DP + Waterfill (PyTorch)
# ==========================================
echo "=========================================="
echo "Experiment 3: Shared Expert DP + Waterfill (PyTorch)"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
SGLANG_USE_TRITON_WATERFILL=0 \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp3_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp3_dp_waterfill_pytorch"

echo ""
echo "Experiment 3 Results:"
extract_metrics "${RESULT_DIR}/exp3_dp_waterfill_pytorch.jsonl"
echo ""

# ==========================================
# Experiment 4: Shared Expert DP + Waterfill (Triton)
# ==========================================
echo "=========================================="
echo "Experiment 4: Shared Expert DP + Waterfill (Triton)"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
SGLANG_USE_TRITON_WATERFILL=1 \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp4_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp4_dp_waterfill_triton"

echo ""
echo "Experiment 4 Results:"
extract_metrics "${RESULT_DIR}/exp4_dp_waterfill_triton.jsonl"
echo ""

# ==========================================
# Experiment 5: Triton + Fake Sync
# ==========================================
echo "=========================================="
echo "Experiment 5: Triton + Fake Sync"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
SGLANG_USE_TRITON_WATERFILL=1 \
SGLANG_FAKE_SYNC_EXPERIMENT=1 \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp5_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp5_dp_waterfill_triton_fake_sync"

echo ""
echo "Experiment 5 Results:"
extract_metrics "${RESULT_DIR}/exp5_dp_waterfill_triton_fake_sync.jsonl"
echo ""

# ==========================================
# Experiment 6: Waterfill Algo + Uniform Dispatch
# ==========================================
echo "=========================================="
echo "Experiment 6: Waterfill Algo + Uniform Dispatch"
echo "=========================================="
kill_server

SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR} \
SGLANG_USE_TRITON_WATERFILL=1 \
SGLANG_FAKE_DISPATCH=1 \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp6_server.log 2>&1 &

wait_for_server
run_benchmark_with_profile "exp6_waterfill_algo_uniform_dispatch"

echo ""
echo "Experiment 6 Results:"
extract_metrics "${RESULT_DIR}/exp6_waterfill_algo_uniform_dispatch.jsonl"
echo ""

# ==========================================
# Summary
# ==========================================
kill_server

echo "=========================================="
echo "                SUMMARY                   "
echo "=========================================="
echo ""
echo "Experiment 1 (TP8 Baseline):"
extract_metrics "${RESULT_DIR}/exp1_tp8_baseline.jsonl"
echo ""
echo "Experiment 2 (DP + Uniform):"
extract_metrics "${RESULT_DIR}/exp2_dp_uniform.jsonl"
echo ""
echo "Experiment 3 (DP + Waterfill - PyTorch):"
extract_metrics "${RESULT_DIR}/exp3_dp_waterfill_pytorch.jsonl"
echo ""
echo "Experiment 4 (DP + Waterfill - Triton):"
extract_metrics "${RESULT_DIR}/exp4_dp_waterfill_triton.jsonl"
echo ""
echo "Experiment 5 (Triton + Fake Sync):"
extract_metrics "${RESULT_DIR}/exp5_dp_waterfill_triton_fake_sync.jsonl"
echo ""
echo "Experiment 6 (Waterfill + Uniform Dispatch):"
extract_metrics "${RESULT_DIR}/exp6_waterfill_algo_uniform_dispatch.jsonl"
echo ""
echo "=========================================="
echo "Torch profiles saved to: ${RESULT_DIR}/"
echo ""
echo "Profile directories:"
ls -la ${RESULT_DIR}/*_profile/ 2>/dev/null || echo "  (no profiles found)"
echo "=========================================="

