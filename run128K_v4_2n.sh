#!/bin/bash
# Weight loading test for DeepSeek V4.1 with Engram support
# Based on run128K_v4.sh but simplified for weight loading only

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# Kill any existing processes
pkill -9 python 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

source /home/w00964934/sglang_deepseek_v4/package/vendors/custom_transformer/bin/set_env.bash
# Use our modified sglang code
export PYTHONPATH=/home/rjw/sglang/python:$PYTHONPATH

# NPU environment
export SGLANG_SET_CPU_AFFINITY=1
export DEEPEP_HCCL_BUFFSIZE=1024
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_BUFFSIZE=400
export ACL_DEVICE_SYNC_TIMEOUT=300
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=enp194s0f0
export GLOO_SOCKET_IFNAME=enp194s0f0

# FIA + MLAPO
export ASCEND_USE_FIA=1
export SGLANG_NPU_USE_MLAPO=1

# DeepEP
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35

# Disable wq_a/wkv fusion - V4.1 weights use separate wq_a and wkv
export SGLANG_OPT_FUSE_WQA_WKV=false

MODEL_PATH=/home/l00890003/codes/weights

IPS=("80.5.17.40" "80.5.17.39")
LOCAL_HOST1=$(hostname -I | awk '{print $1}')
LOCAL_HOST2=$(hostname -I | awk '{print $2}')
export ASCEND_LAUNCH_BLOCKING=1
for i in "${!IPS[@]}"; do
  if [[ "$LOCAL_HOST1" == "${IPS[$i]}" || "$LOCAL_HOST2" == "${IPS[$i]}" ]]; then

  python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
      --served-model-name "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port 30000 \
      --nnodes 2 \
      --node-rank ${i} \
      --tp-size 32 \
      --dp 32 \
      --enable-dp-attention \
      --dist-init-addr 80.5.17.40:6678 \
      --enable-dp-lm-head \
      --disable-cuda-graph \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --max-running-requests 32 \
      --mem-fraction-static 0.75 \
      --quantization modelslim \
      --max-prefill-tokens 2048000 \
      --chunked-prefill-size 24576 \
      --moe-a2a-backend deepep \
      --deepep-mode auto \
      --disable-overlap-schedule \
      --enable-metrics \
      --skip-server-warmup \
      --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}' \
      --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 1}'
      2>&1 | tee /home/l00890003/sglang_v4_loading.log

    exit 1
  fi
done
curl --location 'http://80.5.17.40:6678/generate' --header 'Content-Type: application/json' --data '{
    "text": "The capital of France is",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 8
    }
}'
