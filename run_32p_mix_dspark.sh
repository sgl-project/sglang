#!/bin/bash

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
DRAFT_MODEL_PATH=/home/weights/Kimi-K3-DSpark

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export SGLANG_NPU_PROFILING_BS=4
# export SGLANG_NPU_PROFILING=1
SGLANG_REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SGLANG_NPU_PROFILING_PATH="${SGLANG_NPU_PROFILING_PATH:-${SGLANG_REPO_DIR}/profiling}"
CODEX_SGL_KERNEL_NPU_REPO_DIR=${CODEX_SGL_KERNEL_NPU_REPO_DIR:-/home/hanwlax/test-codes/sgl-kernel-npu}
export PYTHONPATH="${SGLANG_REPO_DIR}/python:${CODEX_SGL_KERNEL_NPU_REPO_DIR}/python/sgl_kernel_npu:${PYTHONPATH:-}"
KDA_COMMIT_MODULE="${CODEX_SGL_KERNEL_NPU_REPO_DIR}/python/sgl_kernel_npu/sgl_kernel_npu/mamba/kda_state_commit.py"
if [[ ! -f "${KDA_COMMIT_MODULE}" ]] || ! grep -q '^def commit_kda_extended_conv_state' "${KDA_COMMIT_MODULE}"; then
    echo "Missing direct KDA state-commit kernel: ${KDA_COMMIT_MODULE}" >&2
    exit 2
fi
# export PYTHONPATH="${SGLANG_REPO_DIR}/python:${PYTHONPATH:-}"

D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
DP_SIZE=${DP_SIZE:-4}
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}
if [[ "${ENABLE_NPU_GRAPH:-1}" == "1" ]]; then
    if [[ -n "${CUDA_GRAPH_BS:-}" ]]; then
        GRAPH_BS_TEXT=${CUDA_GRAPH_BS//,/ }
        read -r -a GRAPH_BS <<< "${GRAPH_BS_TEXT}"
    elif (( (DSPARK_BLOCK_SIZE + 1) % (64 / DP_SIZE) == 0 )); then
        # An aligned verify width can capture/replay the true bs=1 shape.
        GRAPH_BS=(1 2 4 8)
    else
        GRAPH_BS=(2 4 8 16)
    fi
    GRAPH_ARGS=(--cuda-graph-bs "${GRAPH_BS[@]}")
else
    GRAPH_ARGS=(--disable-cuda-graph)
fi
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
        # CANN collective-matmul blocks under SBO side-stream graph capture;
        # serializing it costs more than it saves. Keep #35266's shared MLP
        # overlap as the default until that graph/stream combination is fixed.
        export SGLANG_NPU_FUSED_COLLECTIVE_MATMUL=0
        export SGLANG_NPU_FUSED_RMS_QUANT=1
        export SGLANG_NPU_FUSED_KDA_VERIFY_GATES=${SGLANG_NPU_FUSED_KDA_VERIFY_GATES:-1}
        export SGLANG_NPU_FUSED_KDA_RAGGED_IO=${SGLANG_NPU_FUSED_KDA_RAGGED_IO:-1}
        export SGLANG_NPU_FUSED_KDA_ONORM=${SGLANG_NPU_FUSED_KDA_ONORM:-1}
        export SGLANG_NPU_REUSE_KDA_VERIFY_METADATA=${SGLANG_NPU_REUSE_KDA_VERIFY_METADATA:-1}
        # The 8.50 ms five-run baseline used the regular shared-expert gather.
        # Keep the pre-quantized shared-AG experiment opt-in: it was not part
        # of the validated combination and regressed the critical path.
        export SGLANG_NPU_QUANT_SHARED_AG=${SGLANG_NPU_QUANT_SHARED_AG:-0}
        # Preserve the validated K3 stream-overlap paths. Both paths also
        # check the NPU backend and K3 tensor layout before enabling, so this
        # launch default cannot alter CUDA/GPU execution.
        export SGLANG_NPU_K3_FRONT_OVERLAP=${SGLANG_NPU_K3_FRONT_OVERLAP:-1}
        export SGLANG_NPU_K3_BFA_OVERLAP=${SGLANG_NPU_K3_BFA_OVERLAP:-1}
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}
        # Maximum #35266 + #34944 stack: fused K3 target projection is selected
        # by the model code, while the proposal-side fast paths stay explicit
        # here so an inherited environment cannot silently disable them.
        export SGLANG_DSPARK_FAST_KERNEL=1
        export SGLANG_DSPARK_FUSED_LOCAL_TOP1=${SGLANG_DSPARK_FUSED_LOCAL_TOP1:-1}
        export SGLANG_DSPARK_FAST_SAMPLING=1
        export SGLANG_DSPARK_FOLDED_PROPOSAL=1
        export SGLANG_DSPARK_FOLDED_SAMPLING=1
        export SGLANG_DSPARK_STACKED_CTX_KV=1
        export SGLANG_DSPARK_EMBED_IN_GRAPH=1
        export SGLANG_DSPARK_OPT_MARKOV_W2_BF16=1
        export SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=1
        export SGLANG_DSPARK_ENABLE_MULTI_STREAM=1
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

        echo "K3 launch: tp=64 dp=${DP_SIZE} gamma=${DSPARK_BLOCK_SIZE} graph_bs=${GRAPH_BS[*]:-disabled}"
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
	          --enable-dp-attention --dp-size "${DP_SIZE}" --enable-dp-lm-head \
            --mem-fraction-static 0.75 \
            --max-mamba-cache-size 180 \
            --chunked-prefill-size 16384 \
            "${GRAPH_ARGS[@]}" \
            --reasoning-parser kimi_k3 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
	          --moe-a2a-backend deepep \
            --deepep-mode auto \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}" \
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

python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --num-shots 5 --host 0.0.0.0 --port 30000 --data-path /home/zkk/gsm8k.jsonl

curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --max-concurrency 16 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 16 \
  --disable-ignore-eos \
  --random-range-ratio 1 \
  --warmup-request 0


curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/weights/Kimi-K3-w4a8-int-moe",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 20,
    "temperature": 0
  }'

# 8k_1k_bs1
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
  --extra-request-body '{"routed_dp_rank": 0}' \
  --warmup-request 0 2>&1 | tee "logs/8k_1k_bs1_$(date +%Y-%m-%d_%H-%M-%S).log"

# 8k_1k_bs32
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
  --output-file /home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs/itl_chunk4096_bs32.jsonl \
  --flush-cache 2>&1 | tee "8k_1k_bs32_$(date +%Y-%m-%d_%H-%M-%S).log"

  LOG_DIR=/home/hanwlax/workspace/progress/kimi_k3/bench_serving_logs

# 128k_1k_bs1
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
  --extra-request-body '{"routed_dp_rank": 0}' 2>&1 | tee "128k_1k_bs1_$(date +%Y-%m-%d_%H-%M-%S).log"

# 128k_1k_99cache_bs1
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 0 \
    --gsp-output-len 1 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

# 128k_1k_99cache_bs4
curl --location 'http://0.0.0.0:30000/flush_cache' --header 'Content-Type: application/json'
python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 4 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 0 \
    --gsp-output-len 1 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

python3 -m sglang.bench_serving \
    --dataset-name generated-shared-prefix \
    --backend sglang --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 1 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 4 \
    --gsp-system-prompt-len 127620 \
    --gsp-question-len 1280 \
    --gsp-output-len 1000 \
    --warmup-requests 0 \
    --seed 1 \
    --extra-request-body '{"routed_dp_rank": 0}'

evalscope eval \
    --model /home/weights/Kimi-K3-w4a8-int-moe \
    --api-url http://127.0.0.1:30000/v1 \
    --api-key EMPTY \
    --work-dir "/home/hanwlax/workspace/progress/kimi_k3/gpqa/result_$(date +%Y-%m-%d_%H-%M-%S)" \
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
    --seed 42 2>&1 | tee "gpqa_$(date +%Y-%m-%d_%H-%M-%S).log"

#hot 2
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

ts=$(date +%Y-%m-%d_%H-%M-%S)

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
  --output-file "hit_cache_${ts}.jsonl" \
  2>&1 | tee "hit_cache_${ts}.log"

# profiling
curl --location 'http://127.0.0.1:30000/start_profile' \
  --header 'Content-Type: application/json' \
  --data '{
    "num_steps": 10,
    "profile_by_stage": true,
    "profile_stages": ["decode"],
    "with_stack": false
  }'

curl --location 'http://127.0.0.1:30000/stop_profile'

python3 -m sglang.test.few_shot_gsm8k --num-questions 50 --num-shots 5 --host 0.0.0.0 --port 30000 --data-path /home/zkk/gsm8k.jsonl
