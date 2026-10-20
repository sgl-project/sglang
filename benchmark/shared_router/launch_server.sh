#!/usr/bin/env bash
# Raw serving command; run inside an allocation/container with four isolated GPUs.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 MODEL_PATH accuracy|perf FUSION_0_OR_1 [PORT]" >&2
    exit 2
fi
sr_model=$1
sr_mode=$2
sr_fusion=$3
sr_port=${4:-30000}
[[ "$sr_fusion" == 0 || "$sr_fusion" == 1 ]]
[[ "$sr_port" =~ ^[0-9]+$ ]] && ((sr_port > 1024 && sr_port < 65535))
sr_repo=$(cd "$(dirname "$0")/../.." && pwd)
# Cache locations are allocation-local plumbing, not experimental dispatch flags.
# Preserve them while clearing inherited numerical/performance overrides below.
sr_aiter_cache=${AITER_JIT_DIR:-}
sr_sglang_cache=${SGLANG_JIT_CACHE_DIR:-}
# Prevent inherited experimental flags from silently changing the A/B workload.
while IFS= read -r sr_key; do
    case "$sr_key" in SGLANG_*|AITER_*|DSV41_*|PR86_*) unset "$sr_key";; esac
done < <(compgen -e)
if [[ -n "$sr_aiter_cache" ]]; then export AITER_JIT_DIR="$sr_aiter_cache"; fi
if [[ -n "$sr_sglang_cache" ]]; then export SGLANG_JIT_CACHE_DIR="$sr_sglang_cache"; fi
export PYTHONPATH="$sr_repo/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 SGLANG_SET_CPU_AFFINITY=0 SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1 AITER_FLYDSL_FORCE_REDUCE=1 AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_DSV41_SHARED_ROUTER_FUSION="$sr_fusion"
case "$sr_mode" in
    accuracy) ;;
    perf)
        export SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected
        export SGLANG_RAGGED_VERIFY_MODE=static
        ;;
    *) echo "Mode must be accuracy or perf" >&2; exit 2 ;;
esac
exec python3 -m sglang.launch_server \
    --model-path "$sr_model" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --trust-remote-code --model-impl sglang --json-model-override-args '{"vision_n_layers":0}' \
    --host 127.0.0.1 --port "$sr_port" --nccl-port "$((sr_port + 1))" \
    --tp 4 --ep-size 1 --mem-fraction-static .78 --context-length 1048576 \
    --max-running-requests 128 --chunked-prefill-size 4096 \
    --cuda-graph-max-bs-decode 128 --cuda-graph-backend-prefill breakable \
    --cuda-graph-max-bs-prefill 4096 --schedule-policy fcfs \
    --reasoning-parser auto --tool-call-parser auto --enable-metrics --random-seed 42 \
    --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
    --swa-prefix-tails 3072 --prefill-max-requests 128 --max-prefill-tokens 16384 \
    --disable-radix-cache
