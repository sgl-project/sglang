#!/bin/bash
# DeepEP Waterfill Benchmark Script
#
# Compares DeepEP with and without waterfill load balancing for shared expert
#
# Usage: bash run_deepep_waterfill_benchmark.sh

set -e

MODEL_PATH="/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/"
HOST="0.0.0.0"
PORT=30000
RESULT_DIR="/lustre/raplab/client/xutingz/workspace/bench/deepep_waterfill/$(date +%Y%m%d_%H%M%S)"

# Benchmark parameters
NUM_PROMPTS=512
RANDOM_INPUT=1024
RANDOM_OUTPUT=1
MAX_CONCURRENCY=32
RANDOM_SEED=42  # Fixed seed for reproducibility

mkdir -p ${RESULT_DIR}

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in {1..90}; do
        if curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -q 'DeepSeek-V3'; then
            echo "Server is ready!"
            return 0
        fi
        echo "  Still waiting... ($i/90)"
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

run_benchmark() {
    local name=$1
    local output_file="${RESULT_DIR}/${name}.jsonl"

    echo "Running benchmark: ${name}"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --num-prompts ${NUM_PROMPTS} \
        --random-input ${RANDOM_INPUT} \
        --random-output ${RANDOM_OUTPUT} \
        --seed ${RANDOM_SEED} \
        --max-concurrency ${MAX_CONCURRENCY} \
        --model ${MODEL_PATH} \
        --output-file ${output_file}

    echo "Results saved to: ${output_file}"
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

compare_results() {
    local baseline_file=$1
    local waterfill_file=$2

    python3 -c "
import json

with open('${baseline_file}') as f:
    baseline = json.load(f)
with open('${waterfill_file}') as f:
    waterfill = json.load(f)

baseline_tp = baseline['output_throughput']
waterfill_tp = waterfill['output_throughput']
improvement = (waterfill_tp - baseline_tp) / baseline_tp * 100

baseline_ttft = baseline['mean_ttft_ms']
waterfill_ttft = waterfill['mean_ttft_ms']
ttft_improvement = (baseline_ttft - waterfill_ttft) / baseline_ttft * 100

print(f'Throughput: {baseline_tp:.2f} -> {waterfill_tp:.2f} tok/s ({improvement:+.2f}%)')
print(f'TTFT: {baseline_ttft:.2f} -> {waterfill_ttft:.2f} ms ({ttft_improvement:+.2f}%)')

if waterfill_tp > baseline_tp:
    print('\\n>>> WATERFILL IS FASTER! <<<')
else:
    print('\\n>>> BASELINE IS FASTER <<<')
"
}

echo "=========================================="
echo "DeepEP Waterfill Benchmark"
echo "=========================================="
echo "Parameters:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  NUM_PROMPTS: ${NUM_PROMPTS}"
echo "  RANDOM_INPUT: ${RANDOM_INPUT}"
echo "  RANDOM_OUTPUT: ${RANDOM_OUTPUT}"
echo "  MAX_CONCURRENCY: ${MAX_CONCURRENCY}"
echo "  RANDOM_SEED: ${RANDOM_SEED}"
echo "  RESULT_DIR: ${RESULT_DIR}"
echo ""

# ==========================================
# Experiment 1: DeepEP Baseline (no waterfill)
# ==========================================
echo "=========================================="
echo "Experiment 1: DeepEP Baseline (no waterfill)"
echo "  - moe-a2a-backend: deepep"
echo "  - enable-deepep-waterfill: OFF"
echo "=========================================="
kill_server

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp1_baseline_server.log 2>&1 &

wait_for_server
run_benchmark "exp1_deepep_baseline"

echo ""
echo "Experiment 1 Results:"
extract_metrics "${RESULT_DIR}/exp1_deepep_baseline.jsonl"
echo ""

# ==========================================
# Experiment 2: DeepEP + Waterfill
# ==========================================
echo "=========================================="
echo "Experiment 2: DeepEP + Waterfill"
echo "  - moe-a2a-backend: deepep"
echo "  - enable-deepep-waterfill: ON"
echo "=========================================="
kill_server

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-deepep-waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp2_waterfill_server.log 2>&1 &

wait_for_server
run_benchmark "exp2_deepep_waterfill"

echo ""
echo "Experiment 2 Results:"
extract_metrics "${RESULT_DIR}/exp2_deepep_waterfill.jsonl"
echo ""

# ==========================================
# Experiment 3: DeepEP + Waterfill (Debug Mode)
# ==========================================
echo "=========================================="
echo "Experiment 3: DeepEP + Waterfill (Debug Mode)"
echo "  - SGLANG_DEEPEP_WATERFILL_DEBUG=1"
echo "=========================================="
kill_server

SGLANG_DEEPEP_WATERFILL_DEBUG=1 \
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-deepep-waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp3_waterfill_debug_server.log 2>&1 &

wait_for_server

# Run with fewer prompts for debug
echo "Running benchmark: exp3_deepep_waterfill_debug (fewer prompts for debug)"
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 64 \
    --random-input ${RANDOM_INPUT} \
    --random-output ${RANDOM_OUTPUT} \
    --seed ${RANDOM_SEED} \
    --max-concurrency 8 \
    --model ${MODEL_PATH} \
    --output-file "${RESULT_DIR}/exp3_deepep_waterfill_debug.jsonl"

echo ""
echo "Experiment 3 Results (Debug):"
extract_metrics "${RESULT_DIR}/exp3_deepep_waterfill_debug.jsonl"
echo ""
echo "Debug logs in: ${RESULT_DIR}/exp3_waterfill_debug_server.log"
echo ""

# ==========================================
# Summary
# ==========================================
kill_server

echo "=========================================="
echo "                SUMMARY                   "
echo "=========================================="
echo ""
echo "Experiment 1 (DeepEP Baseline):"
extract_metrics "${RESULT_DIR}/exp1_deepep_baseline.jsonl"
echo ""
echo "Experiment 2 (DeepEP + Waterfill):"
extract_metrics "${RESULT_DIR}/exp2_deepep_waterfill.jsonl"
echo ""

echo "=========================================="
echo "             COMPARISON                   "
echo "=========================================="
compare_results "${RESULT_DIR}/exp1_deepep_baseline.jsonl" "${RESULT_DIR}/exp2_deepep_waterfill.jsonl"
echo ""

echo "=========================================="
echo "All results saved to: ${RESULT_DIR}/"
echo "=========================================="
echo ""
echo "Files:"
ls -la ${RESULT_DIR}/
echo ""
echo "To view server logs:"
echo "  cat ${RESULT_DIR}/exp1_baseline_server.log"
echo "  cat ${RESULT_DIR}/exp2_waterfill_server.log"
echo "  cat ${RESULT_DIR}/exp3_waterfill_debug_server.log"
echo "=========================================="

