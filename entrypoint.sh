#!/bin/bash
set -e

echo "========== Starting entrypoint.sh =========="

# Environment Variables (from LeaderWorkerSet)
echo "LWS_GROUP_SIZE: $LWS_GROUP_SIZE"
echo "LWS_LEADER_ADDRESS: $LWS_LEADER_ADDRESS"
echo "LWS_LEADER_PORT: $LWS_LEADER_PORT"
echo "HOSTNAME: $HOSTNAME"
echo "WORKER_INDEX: $WORKER_INDEX"
echo "AIP_HTTP_PORT: $AIP_HTTP_PORT"

# Echo the user-specified command
echo "SGLang server arguments: $*"

if [[ -z "$LWS_LEADER_ADDRESS" ]]; then
  echo "==> No LWS_LEADER_ADDRESS environment variable detected, assuming single node"
  echo "Starting API server with user-specified arguments..."
  # Start API server with user-defined arguments (using "$@" to forward the command)
  python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port "$AIP_HTTP_PORT" \
    "$@"
# Determine node type based on WORKER_INDEX
elif [[ "$WORKER_INDEX" -eq "0" ]]; then
  echo "==> Leader Node Detected (WORKER_INDEX=0)"
  echo "Starting API server on Leader Node with user-specified arguments..."
  # Start leader with user-defined arguments (using "$@" to forward the command)
  python3 -m sglang.launch_server \
    --dist-init-addr "$LWS_LEADER_ADDRESS:$LWS_LEADER_PORT" \
    --nnodes "$LWS_GROUP_SIZE" \
    --node-rank "$WORKER_INDEX" \
    --host 0.0.0.0 \
    --port "$AIP_HTTP_PORT" \
    "$@"
else
  echo "==> Worker Node Detected (WORKER_INDEX=$WORKER_INDEX)"
  echo "Connecting to leader at: $LWS_LEADER_ADDRESS"

  # Start worker with user-defined arguments (using "$@" to forward the command)
  python3 -m sglang.launch_server \
    --dist-init-addr "$LWS_LEADER_ADDRESS:$LWS_LEADER_PORT" \
    --nnodes "$LWS_GROUP_SIZE" \
    --node-rank "$WORKER_INDEX" \
    "$@"
fi

# Footer
echo "========== entrypoint.sh Complete =========="
