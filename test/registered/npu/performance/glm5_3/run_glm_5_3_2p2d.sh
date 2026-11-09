#!/bin/bash
# GLM-5.3-w8a8 2P2D (2 prefill nodes + 2 decode nodes) launch + bench_serving,
# converted from test_npu_glm_5_3_w8a8_2p2d_32p_in128k_out1k_prefix90_20ms.py.
set -e

export ASCEND_MF_TRANSFER_PROTOCOL="sdma"
# cpu high performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# bind cpu
export SGLANG_SET_CPU_AFFINITY=1
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0
export ASCEND_USE_FIA=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# TODO: fill in the 2 prefill node IPs and 2 decode node IPs.
P_IP=('61.47.19.68' '61.47.19.69')
D_IP=('61.47.19.70' '61.47.19.71')
# TODO: adjust to the local high-speed NIC name.
NIC=enp196s0f0

# pd transfer: point at the first prefill node
export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24670"

MODEL_PATH=/root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.3-w8a8
LOCAL_HOST1=`hostname -I | awk -F " " '{print $1}'`
LOCAL_HOST2=`hostname -I | awk -F " " '{print $2}'`
echo "local ips: ${LOCAL_HOST1} ${LOCAL_HOST2}"

# ===== prefill =====
for i in "${!P_IP[@]}"; do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]]; then
        echo "starting prefill on ${P_IP[$i]} (node-rank $i)"
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export ENABLE_PROFILING=0
        export HCCL_SOCKET_IFNAME=${NIC}
        export GLOO_SOCKET_IFNAME=${NIC}
        export HCCL_BUFFSIZE=1200
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export SGLANG_PP_LAYER_PARTITION=18,20,24,16
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

        python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode prefill \
            --host ${P_IP[$i]} \
            --port 8003 \
            --dist-init-addr ${P_IP[0]}:5000 \
            --disaggregation-bootstrap-port 8998 \
            --trust-remote-code \
            --nnodes 2 \
            --node-rank $i \
            --tp-size 8 \
            --pp-size 4 \
            --dp-size 1 \
            --attn-cp-size 8 \
            --enable-prefill-cp \
            --cp-strategy interleave \
            --enable-dp-attention \
            --enable-dp-lm-head \
            --moe-dense-tp 1 \
            --load-balance-method round_robin \
            --disaggregation-transfer-backend ascend \
            --max-running-requests 64 \
            --served-model-name glm-5 \
            --chunked-prefill-size 4096 \
            --max-prefill-tokens 180000 \
            --moe-a2a-backend deepep \
            --deepep-mode normal \
            --disable-shared-experts-fusion \
            --disable-cuda-graph \
            --dtype bfloat16 \
            --quantization modelslim \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 1 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 2 \
            2>&1 | tee "sglang_prefill_${P_IP[$i]}_$(date +%Y%m%d_%H%M%S).log" &
        break
    fi
done

# ===== decode =====
for i in "${!D_IP[@]}"; do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]]; then
        echo "starting decode on ${D_IP[$i]} (node-rank $i)"
        export HCCL_SOCKET_IFNAME=${NIC}
        export GLOO_SOCKET_IFNAME=${NIC}
        export DEEPEP_HCCL_BUFFSIZE=200
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
        export TASK_QUEUE_ENABLE=0
        export SGLANG_NPU_USE_MULTI_STREAM=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

        python3 -m sglang.launch_server \
            --model-path "${MODEL_PATH}" \
            --disaggregation-mode decode \
            --host ${D_IP[$i]} \
            --port 8004 \
            --dist-init-addr ${D_IP[0]}:5000 \
            --trust-remote-code \
            --nnodes 2 \
            --node-rank $i \
            --tp-size 32 \
            --dp-size 32 \
            --enable-dp-attention \
            --ep-size 32 \
            --mem-fraction-static 0.97 \
            --max-running-requests 32 \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --served-model-name glm-5 \
            --moe-a2a-backend deepep \
            --deepep-mode low_latency \
            --cuda-graph-bs-decode 1 2 3 \
            --disaggregation-transfer-backend ascend \
            --watchdog-timeout 9000 \
            --context-length 180000 \
            --tokenizer-worker-num 4 \
            --disable-shared-experts-fusion \
            --dtype bfloat16 \
            --load-balance-method round_robin \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 4 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 5 \
            2>&1 | tee "sglang_decode_${D_IP[$i]}_$(date +%Y%m%d_%H%M%S).log" &
        break
    fi
done

# ===== bench_serving (run after both P and D are ready) =====
# generated-shared-prefix: 128k shared prefix in / 1k out, 256 prompts, concurrency 64.
python3 -m sglang.bench_serving \
    --backend sglang \
    --host ${D_IP[0]} \
    --port 8004 \
    --model "${MODEL_PATH}" \
    --dataset-name generated-shared-prefix \
    --num-prompts 256 \
    --max-concurrency 64 \
    --input-len 131072 \
    --output-len 1024 \
    --repeat-rate 0.9 \
    --seed 1
