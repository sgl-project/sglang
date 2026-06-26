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
#export SGLANG_NPU_PROFILING_STEP=13


#export ENABLE_PROFILING=1
#export PROFILING_STAGE="decode"
#export PROFILING_STEP=10
#export PROFILING_BS=1


# export HCCL_DETERMINISTIC=true
# export TASK_QUEUE_ENABLE=1

# deepep
# DeepEP MoE dispatch buffer. Must cover the largest batch the benchmark drives:
# with random-input-len=8000, tp/ep=16, the kernel needs ~1759MB (see CamMoeDispatchNormal
# "HCCL_BUFFSIZE is too SMALL"). 1000 crashed the scheduler; give headroom.
export HCCL_BUFFSIZE=2000
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
export PYTHONPATH=/home/gjs/dsv4/sgl_0625/sglang/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Flash-w8a8-mtp-ms

HOST=0.0.0.0          # server bind address (all interfaces)
CLIENT_HOST=127.0.0.1 # address the benchmark/health client connects to (NOT 0.0.0.0)
PORT=30000
#SERVER_LOG=/home/gjs/dsv4/qyb_pr/sglang_server.log
SERVER_LOG=/home/gjs/dsv4/sgl_0625/sglang_server.log

# ---- Launch SGLang server in background ----
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host ${HOST} --port ${PORT} \
    --mem-fraction-static 0.75 \
    --disable-radix-cache --chunked-prefill-size -1 \
    --max-running-requests 128 \
    --disable-overlap-schedule \
    --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --enable-dp-lm-head \
    --kv-cache-dtype auto \
    --cuda-graph-bs 1 2 4 6 \
	--speculative-algorithm NEXTN \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3 \
    2>&1 | tee -a "${SERVER_LOG}" &

# SERVER_PID=$!
# echo "SGLang Server started, PID: ${SERVER_PID}, log: ${SERVER_LOG}"

# # ---- Wait until /health_generate is ready (max ~20 min), fallback to fixed sleep ----
# echo "Waiting server ready..."
# READY=0
# for i in $(seq 1 240); do
#     # bail out early if the server process died
#     if ! kill -0 ${SERVER_PID} 2>/dev/null; then
#         echo "Server process exited unexpectedly. Check ${SERVER_LOG}"
#         exit 1
#     fi
#     if curl -s -o /dev/null -w "%{http_code}" "http://${CLIENT_HOST}:${PORT}/health_generate" 2>/dev/null | grep -q "200"; then
#         READY=1
#         echo "Server is ready after ~$((i*5))s."
#         break
#     fi
#     sleep 5
# done

# if [ "${READY}" -ne 1 ]; then
#     echo "Server did not become ready in time. Check ${SERVER_LOG}"
#     kill ${SERVER_PID} 2>/dev/null
#     exit 1
# fi

# echo "Following server log: ${SERVER_LOG}"
# tail -F "${SERVER_LOG}"

# # ---- Run Serving Benchmark ----
# #BENCH_LOG=/home/gjs/dsv4/qyb_pr/bench_serving.log
# BENCH_LOG=/home/q00886407/dsv4_0609_flash_npu/bench_serving.log
# echo "Running benchmark, results -> ${BENCH_LOG}"
# python3 -m sglang.bench_serving \
#     --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --dataset-name random \
#     --backend sglang \
#     --host ${CLIENT_HOST} \
#     --port ${PORT} \
#     --max-concurrency 64 \
#     --random-input-len 8000 \
#     --random-output-len 1000 \
#     --num-prompts 64 \
#     --disable-ignore-eos \
#     --random-range-ratio 1 \
#     --warmup-request 0 \
#     2>&1 | tee "${BENCH_LOG}"

# # ---- After benchmark, stop server ----
# echo "Benchmark finished, stopping sglang server PID ${SERVER_PID}"
# kill ${SERVER_PID} 2>/dev/null
# wait ${SERVER_PID} 2>/dev/null
# echo "All task done"
