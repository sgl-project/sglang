#!/usr/bin/env bash
set -e
BASE=/home/l/dsv4_flash_pd
MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY='*' no_proxy='*'

/usr/local/Ascend/ascend-toolkit/set_env.sh \
/usr/local/Ascend/nnal/atb/set_env.sh \
/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export PYTHONUNBUFFERED=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32 INF_NAN_MODE_FORCE_DISABLE=1 SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_OPT_FP8_WO_A_GEMM=0 SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1 SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0 SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 USE_NPU_MOE_GATING_TOP_K=1
export SGLANG_RAGGED_VERIFY_MODE=static SGLANG_DSPARK_FAST_KERNEL=0 SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0 SGLANG_DSPARK_QUANT_AUDIT=1 SGLANG_DSPARK_QUANT_AUDIT_STRICT=0
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export ASCEND_MF_STORE_URL=tcp://192.168.25.209:24669
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=1

#!/usr/bin/env bash

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 DEEPEP_HCCL_BUFFSIZE=2600
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192 DEEPEP_NORMAL_LONG_SEQ_ROUND=1
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_ENABLE_WAR_BARRIER=1 SGLANG_FORCE_COARSE_WAR_BARRIER=1 SGLANG_PROFILE_WITH_STACK=False
python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
 --page-size 128 --tp-size 16 --trust-remote-code --device npu --attention-backend dsv4 \
 --watchdog-timeout 9000 --host 192.168.25.209 --port 31000 \
 --disaggregation-mode prefill --disaggregation-transfer-backend ascend --disaggregation-bootstrap-port 18998 \
 --mem-fraction-static 0.62 --prefill-max-requests 256 --max-prefill-tokens 70000 \
 --max-running-requests 256 --chunked-prefill-size 131072 \
 --dp-size 16 --enable-dp-attention --tokenizer-worker-num 16 \
 --moe-a2a-backend deepep --deepep-mode normal --quantization modelslim --enable-dp-lm-head \
 --kv-cache-dtype bfloat16 --disable-cuda-graph --load-balance-method round_robin \
 --enable-metrics --enable-cache-report  --chunked-prefill-size 32768 --init-expert-location /home/l/dsv4_flash_pd/scripts/expert-layout-ais-replicas16.json --ep-dispatch-algorithm dynamic --ep-num-redundant-experts 16 --dp-size 4 --enable-prefill-cp --cp-strategy zigzag --optimistic-prefill-attempts 1 --prefill-max-requests 256 --json-model-override-args '{"use_index_cache":true,"index_topk_freq":2}' --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 512 "$@"
#!/usr/bin/env bash
export HCCL_BUFFSIZE=200 DEEPEP_HCCL_BUFFSIZE=1200 DEEPEP_NORMAL_LONG_SEQ_ROUND=8 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP=1
export SGLANG_NPU_USE_MULTI_STREAM=1
python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
 --page-size 128 --tp-size 16 --trust-remote-code --device npu --attention-backend dsv4 \
 --watchdog-timeout 9000 --host 192.168.25.212 --port 31000 --mem-fraction-static 0.71 \
 --disaggregation-mode decode --disaggregation-transfer-backend ascend --max-running-requests 256 \
 --dp-size 16 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode low_latency \
 --quantization modelslim --enable-dp-lm-head --kv-cache-dtype bfloat16 --load-balance-method round_robin \
 --cuda-graph-bs-decode 1 2 4 7 --speculative-algorithm DSPARK \
 --speculative-draft-model-path "$MODEL_PATH" --speculative-draft-model-quantization modelslim \
 --speculative-draft-attention-backend ascend --speculative-num-draft-tokens 6 \
 --enable-metrics --enable-cache-report  --swa-full-tokens-ratio 0.05 --cuda-graph-bs-decode 1 2 3 4 6 7 8 10 --mem-fraction-static 0.71 --json-model-override-args '{"use_index_cache":true,"index_topk_freq":2}' --tokenizer-worker-num 16 --speculative-dspark-block-size 5 --speculative-num-draft-tokens 6 "$@"
