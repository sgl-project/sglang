#!/bin/bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy

#!/bin/bash
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV

# skip gpu branch
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export USE_NPU_MOE_GATING_TOP_K=1

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0
export SGLANG_DSPARK_QUANT_AUDIT=1
export SGLANG_DSPARK_QUANT_AUDIT_STRICT=0

# path
export PYTHONPATH=/home/w/sglang_dsv4/sglang/python:$PYTHONPATH
LOG_DIR=/home/w/sglang_dsv4/log

MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
#MODEL_PATH=/home/weights/deepseekv4-flash-w8a8-mtp

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

P_IP=('80.5.17.39')
D_IP=('80.5.17.37')

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export ASCEND_MF_STORE_URL="tcp://80.5.17.39:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
# PD分离场景，P节点配置了CP后 D节点不配置CP时 需要配置对应参数
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "Prefill -> ${P_IP[@]}"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
        export DEEPEP_HCCL_BUFFSIZE=2600
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        unset PYTORCH_NPU_ALLOC_CONF
        export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        # export SGLANG_NPU_USE_MULTI_STREAM=1
        # 强制负载均衡
        # export SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=1

        export SGLANG_ENABLE_WAR_BARRIER=1
        export SGLANG_FORCE_COARSE_WAR_BARRIER=1

	    export SGLANG_PROFILE_WITH_STACK=False
        #debug
        export SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=1
            # --ep-dispatch-algorithm static --init-expert-location /home/f/2026-7-30-sglang-deepseekv4-flash/script/eplb/latest.pt \
            # --enable-prefill-cp --cp-strategy interleave \

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 8 \
			      --pp-size 2 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host ${P_IP[$i]} --port 30000 \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port $((8998+$i)) \
            --mem-fraction-static 0.62 \
            --prefill-max-requests 256 \
            --max-prefill-tokens 70000 \
            --max-running-requests 256 \
            --chunked-prefill-size 131072 \
            --dp-size 8 --enable-dp-attention \
            --tokenizer-worker-num 16 \
            --moe-a2a-backend deepep --deepep-mode normal \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --disable-cuda-graph \
	          --load-balance-method round_robin \
            --enable-prefill-cp --cp-strategy interleave \
            --enable-cache-report \
            --enable-hierarchical-cache \
            --hicache-io-backend kernel_ascend \
            --hicache-ratio 1.5 \
            --hicache-write-policy write_through \
            --max-total-tokens 140000 \
            > "${LOG_DIR}/prefill.log" 2>&1
        #exit 1
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_BUFFSIZE=1200
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
        export SGLANG_NPU_USE_MULTI_STREAM=1
        export SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP=1
        export SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=1

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 16 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host 0.0.0.0 --port 30000 \
            --mem-fraction-static 0.71 \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --max-running-requests 256 \
            --dp-size 16 --enable-dp-attention \
            --moe-a2a-backend deepep --deepep-mode low_latency \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --load-balance-method round_robin \
            --cuda-graph-bs-decode 1 2 3 4 5 6\
	          --speculative-algorithm DSPARK \
            --speculative-draft-model-path "${MODEL_PATH}" \
            --speculative-draft-model-quantization modelslim \
            --speculative-draft-attention-backend ascend \
            --speculative-num-draft-tokens 6 \
            > "${LOG_DIR}/decode.log" 2>&1
        # exit 1
    fi
done

# #!/bin/bash
# ### 路由服务 在P节点启动
# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# clear
# unset no_proxy
# cd /home/w-/sglang_dsv4/sglang
# export PYTHONPATH=/home/w/sglang_dsv4/sglang/python:$PYTHONPATH
# # export MODEL_PATH=/home/weights/deepseekv4-flash-w8a8-mtp
# export MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
# export RUST_BACKTRACE=1
# python -m sglang_router.launch_router \
#     --pd-disaggregation --policy cache_aware \
#     --prefill http://80.5.17.39:30000 8998 \
#     --decode http://80.5.17.37:30000 \
#     --host 0.0.0.0 --port 6699 \
#     --queue-timeout-secs 300 \
#     --prometheus-port 9301
