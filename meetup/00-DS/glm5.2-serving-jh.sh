# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEPEP_HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo


MODEL_PATH=/home/weights/GLM-5.2-w4a8
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_HYBRID_DEPLOYMENT=1
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 8 --dp-size 2 \
        --nnodes 1 \
	--disable-shared-experts-fusion \
        --chunked-prefill-size 8192 \
        --max-prefill-tokens 280000 \
        --trust-remote-code \
        --mem-fraction-static 0.90 \
        --served-model-name GLM-5.2-w4a8 \
        --max-running-requests 16 \
	--cuda-graph-max-bs-decode 16 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --load-balance-method round_robin \
        --device npu --host 127.0.0.1 --port 5599

