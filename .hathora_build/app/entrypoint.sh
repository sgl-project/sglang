#!/bin/bash
set -e

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# K2 preset redirect
if [[ "${K2_PRESET:-}" == "1" || "${K2_PRESET:-}" == "true" ]]; then
  log "K2_PRESET enabled, using K2 preset configuration"
  exec /app/k2/preset.sh
fi

# Detect active Infiniband (IB) network interfaces and set NCCL/GLOO ifnames
detect_ib_ifaces() {
  local ifaces=""
  if command -v ibstat >/dev/null 2>&1; then
    # Parse active IB CAs and map to net device names
    for d in $(ibstat | grep -i "Active" -B 8 | grep -E "^CA" | awk '{ print $2 }' | sed "s/'//g"); do
      if [[ -d "/sys/class/infiniband/$d/device/net" ]]; then
        for n in $(ls "/sys/class/infiniband/$d/device/net"); do
          ifaces+="${ifaces:+,}$n"
        done
      fi
    done
  fi

  # Fallback: enumerate IB net interfaces directly
  if [[ -z "$ifaces" && -d /sys/class/infiniband ]]; then
    for d in /sys/class/infiniband/*; do
      [[ -e "$d" ]] || continue
      if [[ -d "$d/device/net" ]]; then
        for n in $(ls "$d/device/net"); do
          ifaces+="${ifaces:+,}$n"
        done
      fi
    done
  fi

  echo "$ifaces"
}

# Configure NCCL/GLOO socket interfaces if not preset
IB_IFACES="$(detect_ib_ifaces)"
if [[ -n "$IB_IFACES" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IB_IFACES}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IB_IFACES}"
  log "Using IB ifaces: $IB_IFACES"
else
  log "No active IB interfaces detected; falling back to defaults"
fi

# Minimal envs aligned with entrypoint_sglang_native.sh
export MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
export TP_SIZE="${TP_SIZE:-1}"
export DTYPE="${DTYPE:-auto}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export QUANTIZATION="${QUANTIZATION:-}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
export SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-1.0}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export ENABLE_METRICS="${ENABLE_METRICS:-true}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
export API_KEY="${API_KEY:-${HATHORA_APP_SECRET:-}}"
export IS_EMBEDDING="${IS_EMBEDDING:-}"
export SPEC_DECODE="${SPEC_DECODE:-}"
export SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-}"
export SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-}"
export DISAGGREGATION_DECODE_TP="${DISAGGREGATION_DECODE_TP:-}"
export DISAGGREGATION_PREFILL_PP="${DISAGGREGATION_PREFILL_PP:-}"
export HF_TOKEN="${HF_TOKEN:-}"

log "Launching Hathora SGLang"
# Optional multi-node (K2) support
if [[ -n "$NNODES" && "$NNODES" -gt 1 ]]; then
  # Determine master IP and node rank via Hathora room config
  if [[ -z "$HATHORA_INITIAL_ROOM_CONFIG" ]]; then
    # Primary node
    MASTER_IP="$HATHORA_PRIVATE_IP"
    export NODE_RANK=0
    # If configured, create a room for the secondary app (K2-B)
    if [[ -n "$KIMI_B_APP_ID" && -n "$HATHORA_TOKEN" ]]; then
      log "Creating secondary room for app $KIMI_B_APP_ID with master_ip=$MASTER_IP"
      python3 /app/create_hathora_room.py "$MASTER_IP" || log "Secondary room creation failed"
    fi
  else
    # Secondary node
    MASTER_IP=$(python3 - <<'PY'
import json, os
cfg = json.loads(os.environ.get('HATHORA_INITIAL_ROOM_CONFIG','{}'))
print(cfg.get('master_ip',''))
PY
)
    export NODE_RANK=${NODE_RANK:-1}
  fi
  export DIST_INIT_ADDR="${DIST_INIT_ADDR:-${MASTER_IP:-127.0.0.1}:20000}"
fi

exec python /app/serve_hathora.py
