#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
DRAFT_MODEL_PATH=/home/weights/Kimi-K3-DSpark
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-2 4 8 16}"

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# profiling
# export SGLANG_NPU_PROFILING=1
export SGLANG_NPU_PROFILING_BS=4
export SGLANG_NPU_PROFILING_PATH="/home/hanwlax/workspace/progress/kimi_k3/profiling"

# A/B
export SGLANG_USE_RECOMPUTE_BEFORE=0  # 优化前算子：1
export SGLANG_KDA_USE_CONV_STATE_TRACK_COPY=1  # 使用算子做COPY：1

# PYTHONPATH
CODEPATH="/home/hanwlax/test-codes/k3"
SGLANG_PATH="${CODEPATH}/sglang/python"
SGLANG_KERNEL_PATH="${CODEPATH}/sgl-kernel-npu/python/sgl_kernel_npu"
export PYTHONPATH="${SGLANG_PATH}:${SGLANG_KERNEL_PATH}:${PYTHONPATH:-}"

D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Mixed -> ${D_IP[$i]}"

        export SGLANG_SET_CPU_AFFINITY=1
        export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
        export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
        export SGLANG_K3_SHARED_EXPERTS_ATTN_TP=1
        export SGLANG_K3_DENSE_MLP_ATTN_TP=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_RAGGED_VERIFY_MODE=static
        export SGLANG_DSPARK_FOLDED_PROPOSAL=0
        export SGLANG_DSPARK_FOLDED_SAMPLING=0
        export SGLANG_DSPARK_STACKED_CTX_KV=0
        export SGLANG_DSPARK_EMBED_IN_GRAPH=0
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        export STREAMS_PER_DEVICE=32
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
        export HCCL_BUFFSIZE=2000
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
        export HCCL_OP_EXPANSION_MODE=AIV
        export HCCL_BUFFSIZE=200
        export DEEPEP_HCCL_BUFFSIZE=1800

        unset ASCEND_CUSTOM_OPP_PATH
        unset SGLANG_NPU_FUSED_MOE_MODE
        unset ENABLE_PROFILING
        unset SGLANG_K3_TRACE_STATE_FILE
        unset SGLANG_K3_TRACE_HIDDEN_FILE
        unset ASCEND_RT_VISIBLE_DEVICES
        unset ASCEND_LAUNCH_BLOCKING

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr 192.168.25.209:5000 --nnodes 4 --node-rank $i \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
	          --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.75 \
            --max-mamba-cache-size 180 \
            --chunked-prefill-size 16384 \
            --cuda-graph-bs $CUDA_GRAPH_BS \
            --reasoning-parser kimi_k3 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
	          --moe-a2a-backend deepep \
            --deepep-mode auto \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \
            --watchdog-timeout 9000  2>&1 | tee "logs/run_32p_mix_$(date +%Y-%m-%d_%H-%M-%S).log"
        exit 1
    fi
done

exit 1


# spec options
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \


LOG_DIR=/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs


# gsm8k
python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --num-shots 5 --host 0.0.0.0 --port 30000 --data-path /home/zkk/gsm8k.jsonl


# single curl
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/weights/Kimi-K3-w4a8-int-moe",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 20,
    "temperature": 0
  }'


# 8k_1k_bs1
date_str=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/8k_1k_bs1_${date_str}"
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --flush-cache \
  --output-details \
  --output-file /home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/8k_1k_bs1_${date_str}.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}' \
  --warmup-request 0 2>&1 | tee "logs/8k_1k_bs1_${date_str}.log"


# 8k_1k_bs32
date_str=$(date +%Y-%m-%d_%H-%M-%S)
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 32 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 32 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0 \
  --output-details \
  --output-file /home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/8k_1k_bs32_${date_str}.jsonl \
  --flush-cache 2>&1 | tee "logs/8k_1k_bs32_${date_str}.log"


# 128k_1k_bs1
date_str=$(date +%Y-%m-%d_%H-%M-%S)
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --seed 42 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0 \
  --flush-cache \
  --output-details \
  --output-file /home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/128k_1k_bs1_${date_str}.jsonl \
  --extra-request-body '{"routed_dp_rank": 0}' 2>&1 | tee "logs/128k_1k_bs1_${date_str}.log"


# 128k_1k_99cache_bs1
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 126720 \
  --random-output-len 1 \
  --num-prompts 1 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --extra-request-body '{"routed_dp_rank": 0}'

date_str=$(date +%Y-%m-%d_%H-%M-%S)
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --cache-report \
  --extra-request-body '{"routed_dp_rank": 0}' \
  --output-details \
  --output-file "128k_1k_bs1_${date_str}.jsonl" \
  2>&1 | tee "logs/128k_1k_bs1_${date_str}.log"


# 128k_1k_99cache_bs4
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 4 \
  --random-input-len 126720 \
  --random-output-len 1 \
  --num-prompts 4 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0

date_str=$(date +%Y-%m-%d_%H-%M-%S)
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 4 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 4 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --cache-report \
  --output-details \
  --output-file "128k_1k_bs1_${date_str}.jsonl" \
  2>&1 | tee "logs/128k_1k_bs1_${date_str}.log"


# 128k_1k_99cache_bs32(on 209)
curl --location 'http://127.0.0.1:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/hanwlax/datasets/shareGPT/sharegpt_natural_shared_128k_32.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 4 \
  --random-input-len 126720 \
  --random-output-len 1 \
  --num-prompts 4 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0

date_str=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/128k_1k_bs32_${date_str}"
python -m sglang.bench_serving \
  --dataset-path /home/hanwlax/datasets/shareGPT/sharegpt_natural_shared_128k_32.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 32 \
  --random-input-len 128000 \
  --random-output-len 1000 \
  --num-prompts 32 \
  --random-range-ratio 1 \
  --seed 42 \
  --warmup-requests 0 \
  --cache-report \
  --output-details \
  --output-file "/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/128k_1k_bs32_${date_str}/details.jsonl" \
  2>&1 | tee "/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/128k_1k_bs32_${date_str}/run.log"


# gpqa
date_str=$(date +%Y-%m-%d_%H-%M-%S)
evalscope eval \
    --model /home/weights/Kimi-K3-w4a8-int-moe \
    --api-url http://127.0.0.1:30000/v1 \
    --api-key EMPTY \
    --work-dir "/home/hanwlax/workspace/progress/kimi_k3/gpqa/result_${date_str}" \
    --no-timestamp \
    --eval-type openai_api \
    --datasets gpqa_diamond \
    --dataset-args '{
      "gpqa_diamond": {
        "local_path": "/home/hanwlax/datasets/gpqa",
        "subset_list": ["gpqa_diamond"],
        "default_subset": "gpqa_diamond"
      }
    }' \
    --generation-config '{
      "max_tokens": 131072,
      "timeout": 10000,
      "temperature": 1.0,
      "top_p": 0.95,
      "extra_body": {
        "reasoning_effort": "max"
      }
    }' \
    --eval-batch-size 32 \
    --seed 42 2>&1 | tee "gpqa_${date_str}.log"


# profiling
curl --location 'http://127.0.0.1:30000/start_profile' \
  --header 'Content-Type: application/json' \
  --data '{
    "output_dir": "/home/hanwlax/workspace/progress/kimi_k3/profiling/0819",
    "num_steps": 10,
    "profile_by_stage": true,
    "with_stack": false,
    "profile_prefix": "128k_1k_bs32_hit_cache_conv_to_track_triton"
  }'

curl --location 'http://127.0.0.1:30000/stop_profile'