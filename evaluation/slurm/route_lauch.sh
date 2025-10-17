#!/bin/bash
set -ex

# Get configuration from parameters
PREFILL_NODE_IP=$1
DECODE_NODE_IP=$2
ROUTER_PORT=${3:-8000}

echo "Starting Router Service on: $(hostname)"
echo "Prefill Node: $PREFILL_NODE_IP"
echo "Decode Node: $DECODE_NODE_IP"
echo "Router Port: $ROUTER_PORT"

# In host network mode, the IP obtained inside the container is the host IP
export MASTER_ADDR=$(hostname -i)

python -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://${PREFILL_NODE_IP}:30000 \
    --decode http://${DECODE_NODE_IP}:30001 \
    --host ${PREFILL_NODE_IP} --port ${ROUTER_PORT} 2>&1 | tee router_log.txt