#!/usr/bin/env bash
set -euo pipefail

export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
if [[ -d "${PYENV_ROOT}/bin" ]]; then
  export PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
  eval "$(pyenv init -)"
  pyenv activate "${PYENV_NAME:-sglang_qianwu}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-/cpfs/user/qianwu/models/note_omni_publish_9800}"
SGL_PORT="${SGL_PORT:-8192}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-393216}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-32}"

MODEL_CONFIG_PATH="${MODEL_PATH}/config.json"
if [[ ! -f "${MODEL_CONFIG_PATH}" ]]; then
  echo "Model config not found: ${MODEL_CONFIG_PATH}" >&2
  exit 1
fi
CHECKPOINT_PRECISION="$("${PYTHON_BIN}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)
quant_config = config.get("quantization_config", config.get("quant_config"))
print("quantized" if quant_config is not None else "bf16")
' "${MODEL_CONFIG_PATH}")"
if [[ "${CHECKPOINT_PRECISION}" == "bf16" ]]; then
  MOE_RUNNER_BACKEND="deep_gemm"
  DEEPEP_DISPATCHER_OUTPUT_DTYPE="bf16"
else
  MOE_RUNNER_BACKEND="auto"
  DEEPEP_DISPATCHER_OUTPUT_DTYPE="auto"
fi
echo "Detected ${CHECKPOINT_PRECISION} checkpoint; MoE runner=${MOE_RUNNER_BACKEND}, DeepEP dispatcher dtype=${DEEPEP_DISPATCHER_OUTPUT_DTYPE}."

export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-8192}"
export SGLANG_MAX_KV_CHUNK_CAPACITY="${SGLANG_MAX_KV_CHUNK_CAPACITY:-8192}"
export NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-0}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
# The first launch JIT-compiles the DeepGEMM/Triton expert kernels independently
# on all eight ranks.  A cold cache can legitimately spend several minutes in a
# single warmup forward, so keep the scheduler watchdog from treating it as a
# deadlock.  Subsequent launches reuse the compiled kernels.
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"
EXTRA_SERVER_ARGS=()
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--disable-radix-cache)
fi
if [[ "${LANGUAGE_ONLY:-0}" == "1" ]]; then
  EXTRA_SERVER_ARGS+=(--language-only)
fi

# Target verification and draft decoding keep separate graph pools. With MTP,
# the default graph range leaves too little headroom for sparse-prefill
# workspaces on an 80 GB GPU.
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  DEEPEP_MODE="${DEEPEP_MODE:-normal}"
  CUDA_GRAPH_ARGS=(
    --cuda-graph-backend-decode disabled
    --cuda-graph-backend-prefill disabled
  )
else
  DEEPEP_MODE="${DEEPEP_MODE:-auto}"
  CUDA_GRAPH_ARGS=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}")
fi

SPECULATIVE_ARGS=()
if [[ "${DISABLE_SPECULATIVE:-0}" != "1" ]]; then
  SPECULATIVE_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS:-3}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    --speculative-draft-model-path "${MODEL_PATH}"
    --speculative-draft-attention-backend fa3
  )
fi

MODEL_OVERRIDE_ARGS='{"im_start_token":"<|img|>","im_token":"<|imgpad|>","im_end_token":"<|endofimg|>","audio_start_token":"<|audio_comp_start|>","audio_token":"<|audio_comp_pad|>","audio_end_token":"<|audio_comp_end|>"}'
if [[ "${DISABLE_DSA:-0}" == "1" ]]; then
  MODEL_OVERRIDE_ARGS='{"im_start_token":"<|img|>","im_token":"<|imgpad|>","im_end_token":"<|endofimg|>","audio_start_token":"<|audio_comp_start|>","audio_token":"<|audio_comp_pad|>","audio_end_token":"<|audio_comp_end|>","index_topk":null}'
fi

# A small SWA pool is sufficient because old SWA states are evictable. Keep
# the saved memory in the full-attention pool so 128K requests fit while
# retaining enough runtime headroom for DeepEP/NVSHMEM.
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --context-length "${CONTEXT_LENGTH}" \
  --enable-dp-attention \
  --dp-size 8 \
  --tp-size 8 \
  --ep-size 8 \
  --port "${SGL_PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.87}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size 16384 \
  --trust-remote-code \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.03}" \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --page-size 64 \
  --moe-dense-tp-size 1 \
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-1800}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  --moe-a2a-backend deepep \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --deepep-dispatcher-output-dtype "${DEEPEP_DISPATCHER_OUTPUT_DTYPE}" \
  --deepep-mode "${DEEPEP_MODE}" \
  --enable-nccl-nvls \
  --enable-multimodal \
  --json-model-override-args "${MODEL_OVERRIDE_ARGS}" \
  --enable-metrics \
  --host 0.0.0.0 \
  "${EXTRA_SERVER_ARGS[@]}"
