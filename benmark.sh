unset http_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
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


# profiling
#export SGLANG_NPU_PROFILING=1
#export SGLANG_NPU_PROFILING_BS=6
#export SGLANG_NPU_PROFILING_STEP=10


# export HCCL_DETERMINISTIC=true
# export TASK_QUEUE_ENABLE=1

# deepep
# DeepEP MoE dispatch buffer. Must cover the largest batch the benchmark drives:
# with random-input-len=8000, tp/ep=16, the kernel needs ~1759MB (see CamMoeDispatchNormal
# "HCCL_BUFFSIZE is too SMALL"). 1000 crashed the scheduler; give headroom.
export HCCL_BUFFSIZE=1000
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

# skip gpu branch
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# Profiling is OFF for benchmark runs (it skews throughput). Set to 1 to capture a trace.
#export SGLANG_NPU_PROFILING=0

# path
#export PYTHONPATH=/home/gjs/dsv4/qyb_pr/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/q00886407/dsv4_0609_flash_npu/sglang/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Flash-w8a8-mtp-ms

HOST=0.0.0.0          # server bind address (all interfaces)
CLIENT_HOST=127.0.0.1 # address the benchmark/health client connects to (NOT 0.0.0.0)
PORT=30000
#SERVER_LOG=/home/gjs/dsv4/qyb_pr/sglang_server.log
SERVER_LOG=/home/q00886407/dsv4_0609_flash_npu/sglang_server.log

# ---- Run Serving Benchmark ----
#BENCH_LOG=/home/gjs/dsv4/qyb_pr/bench_serving.log
BENCH_LOG=/home/q00886407/dsv4_0609_flash_npu/bench_serving.log
echo "Running benchmark, results -> ${BENCH_LOG}"
python3 -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host ${CLIENT_HOST} \
    --port ${PORT} \
    --max-concurrency 96 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 96 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0 \
    2>&1 | tee "${BENCH_LOG}"
