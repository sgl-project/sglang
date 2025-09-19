#!/bin/bash
set -e

# SGLang Hathora Entrypoint Script
# Compatible with the Hathora deployment pattern
# Based on: https://github.com/AndreHathora/llm-serve


log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Set default SGLang configuration if not provided
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
if [[ -n "${TP_SIZE}" ]]; then
  export TP_SIZE
fi
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
export MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-100}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PORT=8000
export ENABLE_METRICS="${ENABLE_METRICS:-false}"
export H100_ONLY="${H100_ONLY:-false}"
export AUTO_USE_FP8_ON_H100="${AUTO_USE_FP8_ON_H100:-false}"
export SPEC_DECODE="${SPEC_DECODE:-}"  # if set to 1/true/yes enables speculative decoding

export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,SHM,P2P,NET}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Reduce NCCL shared segment footprint to fit tiny /dev/shm (64M)
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-1}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-1}"

export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HUGGINGFACE_HUB_TOKEN}" && -n "${HF_TOKEN}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi


log "Starting SGLang Hathora entrypoint script."
log "MODEL_PATH: $MODEL_PATH"
if [[ -n "${TP_SIZE}" ]]; then
  log "TP_SIZE: ${TP_SIZE}"
else
  log "TP_SIZE: auto"
fi
log "MAX_TOTAL_TOKENS: $MAX_TOTAL_TOKENS"
log "MAX_QUEUED_REQUESTS: $MAX_QUEUED_REQUESTS"
log "LOG_LEVEL: $LOG_LEVEL"
log "ENABLE_METRICS: $ENABLE_METRICS"
log "H100_ONLY: $H100_ONLY"
log "AUTO_USE_FP8_ON_H100: $AUTO_USE_FP8_ON_H100"
log "NCCL_SHM_DISABLE: $NCCL_SHM_DISABLE"
log "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
log "NCCL_P2P_LEVEL: $NCCL_P2P_LEVEL"
log "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
log "NCCL_DEBUG: $NCCL_DEBUG"
log "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
log "NCCL_ASYNC_ERROR_HANDLING: $NCCL_ASYNC_ERROR_HANDLING"
log "TORCH_NCCL_ASYNC_ERROR_HANDLING: $TORCH_NCCL_ASYNC_ERROR_HANDLING"
log "NCCL_BUFFSIZE: $NCCL_BUFFSIZE"
log "NCCL_MIN_NCHANNELS: $NCCL_MIN_NCHANNELS"
log "NCCL_MAX_NCHANNELS: $NCCL_MAX_NCHANNELS"
if [[ -n "${SPEC_DECODE}" ]]; then
  log "SPEC_DECODE: ${SPEC_DECODE} (speculative decoding enabled)"
else
  log "SPEC_DECODE: (unset; speculative decoding disabled)"
fi

# Detect GPUs and optionally enforce H100-only
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' || true)
log "Detected GPUs: ${GPU_NAMES}"
if [[ "${H100_ONLY}" == "true" ]]; then
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    log "Non-H100 GPU detected while H100_ONLY=true. Exiting."
    exit 1
  fi
fi

# If multiple GPUs and TP_SIZE is auto, set to number of GPUs up to 8
if [[ -z "${TP_SIZE}" ]]; then
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  if [[ "$NGPU" -gt 0 ]]; then
    if [[ "$NGPU" -gt 8 ]]; then NGPU=8; fi
    export TP_SIZE="$NGPU"
    log "Auto-setting TP_SIZE=${TP_SIZE} based on ${NGPU} GPUs"
  fi
fi

# Extra diagnostics to investigate NCCL/SHM issues
log "==== Diagnostic: /dev/shm usage ===="
df -h /dev/shm || true
cat /proc/mounts | grep shm || true
ls -lah /dev/shm | head -n 50 || true

log "==== Diagnostic: GPU topology (nvidia-smi topo -m) ===="
nvidia-smi topo -m || true

log "==== Diagnostic: NCCL environment dump ===="
env | grep -E '^NCCL_' | sort || true

log "==== Diagnostic: Torch/CUDA/NCCL versions ===="
python - <<'PY' || true
import os, sys
try:
    import torch
    print({
        'torch_version': torch.__version__,
        'cuda_version': getattr(torch.version, 'cuda', None),
        'nccl_available': hasattr(torch.distributed, 'is_nccl_available') and torch.distributed.is_nccl_available(),
        'nccl_version': getattr(getattr(torch, 'cuda', None), 'nccl', None) and getattr(torch.cuda.nccl, 'version', lambda: None)(),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    })
except Exception as e:
    print('torch introspection failed:', e)
print('NCCL env:', {k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
PY

CLEANED_UP=0
function cleanup() {
  if [[ $CLEANED_UP -eq 0 ]]; then
    log "Running cleanup..."
    CLEANED_UP=1
    log "Cleanup complete."
  fi
}

trap cleanup EXIT
trap cleanup ERR
trap cleanup SIGTERM
trap cleanup SIGINT

log "Starting SGLang Hathora FastAPI server..."
log "Server will be available at http://0.0.0.0:${PORT}"
log "OpenAI-compatible endpoint: http://0.0.0.0:${PORT}/v1/chat/completions"
log "Health check endpoint: http://0.0.0.0:${PORT}/health"
log "Metrics endpoint: http://0.0.0.0:${PORT}/metrics"

# Start the SGLang Hathora server
python /app/serve_hathora.py &

SERVER_PID=$!
log "SGLang Hathora server started with PID: $SERVER_PID"

# Wait for the server process
wait $SERVER_PID

log "SGLang Hathora server exited. Entrypoint script ending."
