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
export TP_SIZE="${TP_SIZE:-1}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
export MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-100}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PORT=8000
export ENABLE_METRICS="${ENABLE_METRICS:-false}"
export H100_ONLY="${H100_ONLY:-false}"
export AUTO_USE_FP8_ON_H100="${AUTO_USE_FP8_ON_H100:-true}"


log "Starting SGLang Hathora entrypoint script."
log "MODEL_PATH: $MODEL_PATH"
log "TP_SIZE: $TP_SIZE"
log "MAX_TOTAL_TOKENS: $MAX_TOTAL_TOKENS"
log "MAX_QUEUED_REQUESTS: $MAX_QUEUED_REQUESTS"
log "LOG_LEVEL: $LOG_LEVEL"
log "ENABLE_METRICS: $ENABLE_METRICS"
log "H100_ONLY: $H100_ONLY"
log "AUTO_USE_FP8_ON_H100: $AUTO_USE_FP8_ON_H100"

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
if [[ -z "${TP_SIZE_USER_SET}" && "${TP_SIZE}" == "auto" ]]; then
  NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  if [[ "$NGPU" -gt 0 ]]; then
    if [[ "$NGPU" -gt 8 ]]; then NGPU=8; fi
    export TP_SIZE="$NGPU"
    log "Auto-setting TP_SIZE=${TP_SIZE} based on ${NGPU} GPUs"
  fi
fi




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
