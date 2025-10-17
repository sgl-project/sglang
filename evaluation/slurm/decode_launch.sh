#!/bin/bash
set -ex

# Get configuration from parameters
MODEL_PATH=$1
IBDEVICES=$2
HOST_IP=$3
PORT=${4:-30001}

echo "Starting Decode Service on: $(hostname)"
echo "Model: $MODEL_PATH"
echo "Host IP: $HOST_IP"
echo "Port: $PORT"
echo "IB Devices: $IBDEVICES"

# Set environment variables
export MODEL_PATH=$MODEL_PATH
export IBDEVICES=$IBDEVICES
export host_ip=$HOST_IP

# Use HOST_IP from parameters instead of obtaining it inside the container
python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --disaggregation-mode decode --disaggregation-ib-device ${IBDEVICES} \
    --host ${HOST_IP} --port ${PORT} --trust-remote-code \
    --tp 8 2>&1 | tee server_decode_log.txt