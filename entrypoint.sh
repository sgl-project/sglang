#!/bin/bash
set -e

# SGLang Hathora Entrypoint Script
# Compatible with the Hathora deployment pattern
# Based on: https://github.com/AndreHathora/llm-serve

# Required env vars: HATHORA_HOSTNAME, HATHORA_DEFAULT_PORT, HATHORA_REGION
GCP_PROJECT=hathora-dev-1ba2

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Environment validation
if [[ -z "$GCP_SERVICE_ACCOUNT_KEY_BASE64" ]]; then
  echo "GCP_SERVICE_ACCOUNT_KEY_BASE64 is not set. Exiting." 
  exit 1
fi
if [[ -z "$HATHORA_HOSTNAME" ]]; then
  echo "HATHORA_HOSTNAME is not set. Exiting."
  exit 1
fi
if [[ -z "$HATHORA_DEFAULT_PORT" ]]; then
  echo "HATHORA_DEFAULT_PORT is not set. Exiting."
  exit 1
fi
if [[ -z "$HATHORA_REGION" ]]; then
  echo "HATHORA_REGION is not set. Exiting."
  exit 1
fi

# Set default SGLang configuration if not provided
export MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export TP_SIZE="${TP_SIZE:-1}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PORT="${HATHORA_DEFAULT_PORT:-8000}"
export ENABLE_METRICS="${ENABLE_METRICS:-true}"
export H100_ONLY="${H100_ONLY:-true}"
export AUTO_USE_FP8_ON_H100="${AUTO_USE_FP8_ON_H100:-true}"

log "Decoding service account key from environment variable..."
# Decode base64 string and save to temporary JSON file
echo "$GCP_SERVICE_ACCOUNT_KEY_BASE64" | base64 -d > /tmp/gcp-sa-key.json
# Set environment variable to point to the decoded service account key file
gcloud auth activate-service-account --key-file=/tmp/gcp-sa-key.json
gcloud config set project $GCP_PROJECT

log "Starting SGLang Hathora entrypoint script."
log "HATHORA_HOSTNAME: $HATHORA_HOSTNAME"
log "HATHORA_DEFAULT_PORT: $HATHORA_DEFAULT_PORT"
log "HATHORA_REGION: $HATHORA_REGION"
log "GCP_PROJECT: $GCP_PROJECT"
log "MODEL_PATH: $MODEL_PATH"
log "TP_SIZE: $TP_SIZE"
log "MAX_TOTAL_TOKENS: $MAX_TOTAL_TOKENS"
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

log "Detecting GCP zone and VM name from public IP..."
GCP_VM_PUBLIC_IP=$(dig +short $HATHORA_HOSTNAME)
GCP_ZONE=$(gcloud compute instances list --filter="networkInterfaces[0].accessConfigs[0].natIP=$GCP_VM_PUBLIC_IP" --format="value(zone)")
GCP_VM_NAME=$(gcloud compute instances list --filter="networkInterfaces[0].accessConfigs[0].natIP=$GCP_VM_PUBLIC_IP" --format="value(name)")
GCP_NEG_NAME="llm-serve-$GCP_ZONE"
log "Detected GCP_VM_PUBLIC_IP: $GCP_VM_PUBLIC_IP"
log "Detected GCP_ZONE: $GCP_ZONE"
log "Detected GCP_VM_NAME: $GCP_VM_NAME"
log "GCP_NEG_NAME: $GCP_NEG_NAME"

function register_endpoint() {
  log "Registering endpoint instance=$GCP_VM_NAME,port=$HATHORA_DEFAULT_PORT to NEG $GCP_NEG_NAME in $GCP_ZONE..."
  local start_time=$(date +%s)
  gcloud compute network-endpoint-groups update \
    "$GCP_NEG_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --add-endpoint="instance=$GCP_VM_NAME,port=$HATHORA_DEFAULT_PORT"
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log "Endpoint registered. Took ${duration}s."
}

function unregister_endpoint() {
  log "Unregistering endpoint instance=$GCP_VM_NAME,port=$HATHORA_DEFAULT_PORT from NEG $GCP_NEG_NAME in $GCP_ZONE..."
  local start_time=$(date +%s)
  gcloud compute network-endpoint-groups update \
    "$GCP_NEG_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --remove-endpoint="instance=$GCP_VM_NAME,port=$HATHORA_DEFAULT_PORT"
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log "Endpoint unregistered. Took ${duration}s."
}

CLEANED_UP=0
function cleanup() {
  if [[ $CLEANED_UP -eq 0 ]]; then
    log "Running cleanup..."
    unregister_endpoint
    CLEANED_UP=1
    log "Cleanup complete."
  fi
}

trap cleanup EXIT
trap cleanup ERR
trap cleanup SIGTERM
trap cleanup SIGINT

register_endpoint

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
