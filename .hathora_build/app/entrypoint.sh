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
export ALLOW_CONTINUE_ON_LOW_SHM="${ALLOW_CONTINUE_ON_LOW_SHM:-1}"

export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,SHM,P2P,NET}"
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
log "AUTO_USE_FP8_ON_H100: $AUTO_USE_FP8_ON_H100"
if [[ -n "${CONCURRENCY}" ]]; then
  log "CONCURRENCY: ${CONCURRENCY}"
fi
if [[ -n "${MAX_CONCURRENCY}" ]]; then
  log "MAX_CONCURRENCY: ${MAX_CONCURRENCY}"
fi
if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
  log "MAX_RUNNING_REQUESTS: ${MAX_RUNNING_REQUESTS}"
fi
if [[ -n "${CUDA_GRAPH_MAX_BS}" ]]; then
  log "CUDA_GRAPH_MAX_BS: ${CUDA_GRAPH_MAX_BS}"
fi
log "NCCL_SHM_DISABLE: $NCCL_SHM_DISABLE"
log "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"
log "NCCL_P2P_LEVEL: $NCCL_P2P_LEVEL"
log "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
log "NCCL_DEBUG: $NCCL_DEBUG"
log "NCCL_DEBUG_SUBSYS: $NCCL_DEBUG_SUBSYS"
log "TORCH_NCCL_ASYNC_ERROR_HANDLING: $TORCH_NCCL_ASYNC_ERROR_HANDLING"
log "NCCL_BUFFSIZE: $NCCL_BUFFSIZE"
log "NCCL_MIN_NCHANNELS: $NCCL_MIN_NCHANNELS"
log "NCCL_MAX_NCHANNELS: $NCCL_MAX_NCHANNELS"
if [[ -n "${SPEC_DECODE}" ]]; then
  log "SPEC_DECODE: ${SPEC_DECODE} (speculative decoding enabled)"
else
  log "SPEC_DECODE: (unset; speculative decoding disabled)"
fi

# Auto-detect TP early if not set, to make guards accurate
if [[ -z "${TP_SIZE}" ]]; then
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  if [[ "${NGPU}" -gt 0 ]]; then
    if [[ "${NGPU}" -ge 8 ]]; then TP_CHOSEN=8
    elif [[ "${NGPU}" -ge 4 ]]; then TP_CHOSEN=4
    elif [[ "${NGPU}" -ge 2 ]]; then TP_CHOSEN=2
    else TP_CHOSEN=1
    fi
    export TP_SIZE="${TP_CHOSEN}"
    log "Auto-setting TP_SIZE=${TP_SIZE} based on ${NGPU} GPUs"
  fi
fi

# Guard: detect small /dev/shm and fail-fast or auto-degrade; scale requirement by effective world size
SHM_BYTES=$(df -B1 /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [[ -z "${SHM_BYTES}" ]]; then SHM_BYTES=0; fi

# Read PD disaggregation decode TP if provided (support two env spellings)
DISAGG_DECODE_TP_VAL="${DISAGGREGATION_DECODE_TP:-${DISAGG_DECODE_TP:-}}"
if [[ -z "${DISAGG_DECODE_TP_VAL}" ]]; then DISAGG_DECODE_TP_VAL=1; fi

TP_CUR=${TP_SIZE:-1}
EFFECTIVE_TP=${TP_CUR}
if [[ "${DISAGG_DECODE_TP_VAL}" =~ ^[0-9]+$ ]] && [[ "${DISAGG_DECODE_TP_VAL}" -gt "${EFFECTIVE_TP}" ]]; then
  EFFECTIVE_TP=${DISAGG_DECODE_TP_VAL}
fi

# Clamp effective TP to supported set {1,2,4,8}
if [[ "${EFFECTIVE_TP}" -ge 8 ]]; then EFFECTIVE_TP=8
elif [[ "${EFFECTIVE_TP}" -ge 4 ]]; then EFFECTIVE_TP=4
elif [[ "${EFFECTIVE_TP}" -ge 2 ]]; then EFFECTIVE_TP=2
else EFFECTIVE_TP=1
fi

# Defaults per parallel size; overridable via env
default_bytes_for() {
  case "$1" in
    1) echo $((256 * 1024 * 1024)) ;;
    2) echo $((512 * 1024 * 1024)) ;;
    4) echo $((1024 * 1024 * 1024)) ;;
    8) echo $((2 * 1024 * 1024 * 1024)) ;;
    *) echo $((2 * 1024 * 1024 * 1024)) ;;
  esac
}

override_bytes_env="MIN_SHM_BYTES_TP_${EFFECTIVE_TP}"
MIN_REQUIRED=$(default_bytes_for "${EFFECTIVE_TP}")
eval "OVERRIDE_VAL=\${${override_bytes_env}:-}"
if [[ -n "${OVERRIDE_VAL}" ]]; then MIN_REQUIRED=${OVERRIDE_VAL}; fi

# If too small, decide action
if [[ "${EFFECTIVE_TP}" -ge 2 && "${SHM_BYTES}" -lt "${MIN_REQUIRED}" ]]; then
  log "WARN: /dev/shm too small for effective parallel size=${EFFECTIVE_TP}."
  log "  Detected /dev/shm: ${SHM_BYTES} bytes; required: ${MIN_REQUIRED} bytes."

  ALLOW_CONTINUE_ON_LOW_SHM_LC=$(echo "${ALLOW_CONTINUE_ON_LOW_SHM:-}" | tr '[:upper:]' '[:lower:]')

  if [[ "${ALLOW_CONTINUE_ON_LOW_SHM_LC}" =~ ^(1|true|yes)$ ]]; then
    log "Continuing with TP_SIZE=${TP_CUR}, effective_parallel=${EFFECTIVE_TP}; NCCL may fail under load."
  else
    export TP_SIZE=1
    EFFECTIVE_TP=1
    log "Auto-degrading TP_SIZE to 1 due to small /dev/shm. Set ALLOW_CONTINUE_ON_LOW_SHM=1 to skip."
  fi
fi

# Heuristic NCCL tuning by available shm
channels=1
buffsize=$((1 * 1024 * 1024))
if [[ "${SHM_BYTES}" -ge $((2 * 1024 * 1024 * 1024)) ]]; then
  channels=4; buffsize=$((4 * 1024 * 1024))
elif [[ "${SHM_BYTES}" -ge $((1024 * 1024 * 1024)) ]]; then
  channels=2; buffsize=$((2 * 1024 * 1024))
fi
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-${channels}}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-${channels}}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-${buffsize}}
log "NCCL tuning: channels=${channels}, buffsize=${buffsize} bytes, shm=${SHM_BYTES} bytes"

# Detect GPUs and optionally enforce H100-only
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' || true)
log "Detected GPUs: ${GPU_NAMES}"
:


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
