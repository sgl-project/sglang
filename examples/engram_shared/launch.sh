#!/usr/bin/env bash
# One vLLM owner with SGLang P TP2 / D TP2 and the native NIXL router.
set -euo pipefail

role=${1:?usage: launch.sh owner|prefill|decode|router}
if [[ "$role" != router ]]; then
  : "${MODEL_DIR:?set MODEL_DIR to the local DeepSeek-V4.1-Flash checkpoint}"
  : "${TOKEN_FILE:?set TOKEN_FILE to a private token file shared with consumers}"
  if [[ ! -r "$TOKEN_FILE" ]]; then
    echo "TOKEN_FILE is not readable: $TOKEN_FILE" >&2
    exit 2
  fi
fi

sglang_python=${SGLANG_PYTHON:-python}
owner_endpoint=${OWNER_ENDPOINT:-127.0.0.1:56000}
owner_port=${owner_endpoint##*:}

run() {
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    exec "$@"
  fi
}

case "$role" in
  owner)
    run "${VLLM_PYTHON:-python}" \
      -m vllm.models.deepseek_v41.nvidia.ops.shared_engram_owner \
      --model-dir "$MODEL_DIR" \
      --listen-host "${OWNER_BIND_HOST:-127.0.0.1}" \
      --listen-port "$owner_port" \
      --token-file "$TOKEN_FILE" --device "${OWNER_DEVICE:-0}"
    ;;
  prefill|decode)
    : "${GENERATION:?copy generation from ENGRAM_OWNER_READY line}"
    export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
    export SGLANG_OPT_USE_TILELANG_MHC_POST=0
    export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp}
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
    if [[ "$role" == prefill ]]; then
      port=${P_PORT:-31000}
      base_gpu=${P_BASE_GPU_ID:-0}
      nccl_port=${P_NCCL_PORT:-30000}
    else
      port=${D_PORT:-31001}
      base_gpu=${D_BASE_GPU_ID:-2}
      nccl_port=${D_NCCL_PORT:-30100}
    fi
    engram_json=$(OWNER_ENDPOINT="$owner_endpoint" GENERATION="$GENERATION" \
      TOKEN_FILE="$TOKEN_FILE" "$sglang_python" -c \
      'import json, os; print(json.dumps({"placement": "external", "shared_host_endpoint": os.environ["OWNER_ENDPOINT"], "shared_host_token_file": os.environ["TOKEN_FILE"], "shared_host_generation": os.environ["GENERATION"]}))')
    run "$sglang_python" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --host "${SERVE_HOST:-127.0.0.1}" \
      --port "$port" --tp 2 --base-gpu-id "$base_gpu" \
      --nccl-port "$nccl_port" --ep-size 1 --language-model-only \
      --disaggregation-mode "$role" --disaggregation-transfer-backend nixl \
      --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-31500}" \
      --context-length 32768 --mem-fraction-static 0.90 \
      --max-running-requests 2 --chunked-prefill-size 8192 \
      --load-format safetensors \
      --model-loader-extra-config '{"enable_multithread_load":false}' \
      --bf16-gemm-backend torch --fp8-gemm-backend triton \
      --engram-config "$engram_json" --disable-cuda-graph \
      --disable-flashinfer-autotune --watchdog-timeout 3600
    ;;
  router)
    run "$sglang_python" -m sglang_router.launch_router \
      --pd-disaggregation --mini-lb \
      --prefill "${P_URL:-http://127.0.0.1:31000}" \
      --decode "${D_URL:-http://127.0.0.1:31001}" \
      --host "${ROUTER_HOST:-127.0.0.1}" --port "${ROUTER_PORT:-31002}"
    ;;
  *)
    echo "usage: launch.sh owner|prefill|decode|router" >&2
    exit 2
    ;;
esac
