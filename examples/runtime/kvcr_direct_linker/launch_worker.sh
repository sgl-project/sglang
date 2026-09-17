#!/usr/bin/env bash
# Launch one SGLang worker with the KVCR direct linker.
#
# Usage:
#   launch_worker.sh --model <path> [--port 30000] [--control-port 25000]
#                    [--dram-gib 8] [--advertise-host <ip>] [--tp 1]
#                    [--page-size 64] [--mem-fraction 0.8] [-- <extra sglang args>]
set -euo pipefail

MODEL=""
PORT=30000
CONTROL_PORT=25000
DRAM_GIB=8
ADVERTISE_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
TP=1
PAGE_SIZE=64
MEM_FRACTION=0.8
DEADLINE_MS=2000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --control-port) CONTROL_PORT="$2"; shift 2 ;;
    --dram-gib) DRAM_GIB="$2"; shift 2 ;;
    --advertise-host) ADVERTISE_HOST="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --page-size) PAGE_SIZE="$2"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="$2"; shift 2 ;;
    --deadline-ms) DEADLINE_MS="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "--model is required" >&2; exit 2; }

DRAM_BYTES=$((DRAM_GIB * 1024 * 1024 * 1024))
EXTRA_CONFIG=$(cat <<EOF
{"local_dram_bytes_per_worker": ${DRAM_BYTES},
 "enable_remote_hint": true,
 "control_port": ${CONTROL_PORT},
 "control_advertise_host": "${ADVERTISE_HOST}",
 "preparation_deadline_ms": ${DEADLINE_MS}}
EOF
)

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" \
  --tp-size "$TP" \
  --page-size "$PAGE_SIZE" \
  --mem-fraction-static "$MEM_FRACTION" \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config "$EXTRA_CONFIG" \
  "$@"
