#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
MODE=${LIM_EVAL_MODE:?}
TAG=${LIM_EVAL_TAG:?}
exec >"/task/logs/${TAG}.server.log" 2>&1
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
unset SGLANG_ENABLE_SPEC_V2 SGLANG_NPU_LIM_LIBRARY VLLM_ASCEND_SRC CUSTOM_OP_ROOT
unset ASCEND_RT_VISIBLE_DEVICES
export SGLANG_SET_CPU_AFFINITY=1 STREAMS_PER_DEVICE=32
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export HCCL_BUFFSIZE=1000 HCCL_OP_EXPANSION_MODE=AIV
# DeepEP auto shares one HCCL window between normal and V2 low-latency ops.
# Keep their payload and synchronization state in the hybrid window layout.
export DEEPEP_HYBRID_DEPLOYMENT=1
# Bound V2's allocated receive capacity for this launcher's decode batch <= 16.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
export TRANSFORMERS_VERBOSITY=error
export SGLANG_NPU_PROFILING=0 SGLANG_NPU_PROFILING_BS=16
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export TORCHINDUCTOR_CACHE_DIR=/task/cache/inductor
export TRITON_CACHE_DIR=/task/cache/triton
export SGLANG_NPU_SPARSE_KV_CACHE_SIZE=2048
export SGLANG_NPU_SPARSE_KV_INDEXER=legacy
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=0
export SGLANG_NPU_ENABLE_COPY_SFA=0
if [ "$MODE" = lim_copy_sfa ]; then
  export SGLANG_NPU_SPARSE_KV_INDEXER=lim
  export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
  export SGLANG_NPU_ENABLE_COPY_SFA=1
elif [ "$MODE" != baseline ]; then
  echo "Unknown mode: $MODE" >&2; exit 2
fi
echo $$ >"/task/results/${TAG}.server-pid"
exec python3 -m sglang.launch_server \
  --model-path /models/GLM-5.2-w4a8 --served-model-name GLM-5.2-w4a8 \
  --attention-backend ascend --device npu --tp-size 16 --ep-size 16 \
  --nnodes 1 --attn-dp-size 2 --chunked-prefill-size 2048 \
  --max-prefill-tokens 32768 --trust-remote-code --mem-fraction-static 0.82 \
  --cuda-graph-bs-decode 8 --max-running-requests 16 \
  --quantization modelslim --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --moe-a2a-backend deepep --deepep-mode auto --load-balance-method round_robin \
  --disable-shared-experts-fusion --disable-radix-cache \
  --context-length 36864 --max-total-tokens 131072 --page-size 128 \
  --random-seed 42 --host 127.0.0.1 --port 8810
