#!/usr/bin/env bash
set -euo pipefail

export PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
if [[ -d "${PYENV_ROOT}/bin" ]]; then
  export PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
  eval "$(pyenv init -)"
  pyenv activate "${PYENV_NAME:-sglang_qianwu}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-/diancpfs/user/xiaoke/ckpts/dots/release/dots3_note_omni_sft_stage1_dsa_384k_release_fixnan/iter_0009800_hf_fp8_infra}"
SGL_PORT="${SGL_PORT:-8192}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

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

CUDA_GRAPH_ARGS=()
if [[ "${ENABLE_CUDA_GRAPH:-0}" != "1" ]]; then
  CUDA_GRAPH_ARGS+=(--disable-cuda-graph)
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

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --context-length "${CONTEXT_LENGTH}" \
  --enable-dp-attention \
  --dp-size 8 \
  --tp-size 8 \
  --ep-size 8 \
  --port "${SGL_PORT}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}" \
  --max-running-requests 256 \
  --chunked-prefill-size 16384 \
  --trust-remote-code \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.2}" \
  --prefill-attention-backend fa3 \
  --decode-attention-backend fa3 \
  --page-size 64 \
  --moe-dense-tp-size 1 \
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-1800}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-nccl-nvls \
  --enable-multimodal \
  --json-model-override-args "${MODEL_OVERRIDE_ARGS}" \
  --enable-metrics \
  --host 0.0.0.0 \
  "${EXTRA_SERVER_ARGS[@]}"
