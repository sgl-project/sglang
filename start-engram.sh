#!/bin/bash
# Weight loading test for DeepSeek V4.1 with Engram support

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# Kill any existing processes
pkill -9 python 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
# Use our modified sglang code
export PYTHONPATH=`pwd`/python:$PYTHONPATH

# NPU environment
export SGLANG_SET_CPU_AFFINITY=1
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_BUFFSIZE=400
export ACL_DEVICE_SYNC_TIMEOUT=300
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_OPT_ENGRAM_HOST_OFFLOAD=1

# FIA + MLAPO
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1

# DeepEP
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35

# Disable wq_a/wkv fusion - V4.1 weights use separate wq_a and wkv
export SGLANG_OPT_FUSE_WQA_WKV=false

#MODEL_PATH=/data/ascend-ci-share-pkking-sglang/Aurora-W8A8-144/Aurora-W8A8-144
MODEL_PATH=/home/l00890003/codes/weights
SERVER_HOST=80.5.17.40
SERVER_PORT=6688

# Timestamped log file: logs/engram.YYYYmmdd_HHMMSS.log
mkdir -p logs
LOG_FILE=logs/engram.$(date +%Y%m%d_%H%M%S).log
echo "logging to ${LOG_FILE}"

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --served-model-name "${MODEL_PATH}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 16 \
    --dp-size 2 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --disable-cuda-graph \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --watchdog-timeout 9000 \
    --max-running-requests 3 \
    --mem-fraction-static 0.8 \
    --quantization modelslim \
    --max-prefill-tokens 204800 \
    --chunked-prefill-size 24576 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disable-overlap-schedule \
    --enable-metrics \
    --skip-server-warmup \
    --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"], "num_hidden_layers": 6}' \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 1}' \
    2>&1 | tee "${LOG_FILE}"

