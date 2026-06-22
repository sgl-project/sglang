#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4 model directory}"

HOST=${HOST:-0.0.0.0}
CLIENT_HOST=${CLIENT_HOST:-127.0.0.1}
PORT=${PORT:-30000}
SERVER_LOG=${SERVER_LOG:-/tmp/sglang_dsv4_server.log}
READY_TIMEOUT=${READY_TIMEOUT:-1200}
TP_SIZE=${TP_SIZE:-16}
DP_SIZE=${DP_SIZE:-16}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}

for env_file in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
    fi
done

if [[ "${TUNE_HOST:-0}" == "1" ]]; then
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
fi

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export INF_NAN_MODE_FORCE_DISABLE=${INF_NAN_MODE_FORCE_DISABLE:-1}
export SGLANG_SET_CPU_AFFINITY=${SGLANG_SET_CPU_AFFINITY:-1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}
export DEEP_NORMAL_MODE_USE_INT8_QUANT=${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}
export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-16}
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-2048}
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-False}
export SGLANG_OPT_FUSE_WQA_WKV=${SGLANG_OPT_FUSE_WQA_WKV:-0}
export SGLANG_OPT_USE_FUSED_HASH_TOPK=${SGLANG_OPT_USE_FUSED_HASH_TOPK:-False}
export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-1}
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
export SGLANG_NPU_PROFILING=${SGLANG_NPU_PROFILING:-0}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size "${TP_SIZE}" \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host "${HOST}" \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --disable-overlap-schedule \
    --dp-size "${DP_SIZE}" \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --quantization modelslim \
    --enable-dp-lm-head \
    --kv-cache-dtype auto \
    --cuda-graph-bs 1 2 4 6 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3 \
    2>&1 | tee -a "${SERVER_LOG}" &
SERVER_PID=$!

echo "SGLang server started, PID=${SERVER_PID}, log=${SERVER_LOG}"
deadline=$((SECONDS + READY_TIMEOUT))
until curl --fail --silent "http://${CLIENT_HOST}:${PORT}/health_generate" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited unexpectedly. Check ${SERVER_LOG}." >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Server did not become ready within ${READY_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 5
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLIENT_HOST=${CLIENT_HOST} PORT=${PORT} "${SCRIPT_DIR}/benchmark.sh"
