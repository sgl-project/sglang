---
title: "DeepSeek-V4-Flash"
metatags:
  description: "Best Practice for DeepSeek-V4-Flash on Ascend NPU"
---

<Note>
This page focuses on optimal configuration and benchmark results for DeepSeek-V4-Flash on the Ascend NPU. For environment setup, model weight download, feature configuration, and deployment instructions, etc., see the [DeepSeek-V4-Flash Model Tutorial](/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash).

On the A3 Series, each card has 2 dies, so `--tp-size` is twice the card count; see [Ascend NPU Reference](/docs/hardware-platforms/ascend-npus/reference/glossary#hardware) for details.
</Note>

### High Throughput

| Model | Hardware | Cards | Deploy Mode | Dataset | TPOT | Quantization | Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | Ascend A3 Series Products | 16 | PD Disaggregation | 8k+1k | 50ms | W8A8 INT8 | [Optimal Configuration](#deepseek-v4-flash-w8a8-1p1d-16p-in8k-out1k-50ms) |
| DeepSeek-V4-Flash | Ascend A3 Series Products | 8 | PD Mixed | 8k+1k | 50ms | W8A8 INT8 | [Optimal Configuration](#deepseek-v4-flash-w8a8-8p-in8k-out1k-50ms) |

## Optimal Configuration

<a id="pd-disaggregation" title="Referenced by external docs. Verify before removing."></a>

### DeepSeek-V4-Flash W8A8 1P1D 16P IN8K OUT1K 50ms

**Model**: DeepSeek-V4-Flash

**Hardware**: Ascend A3 Series Products

**Cards**: 16

**Deploy Mode**: PD Disaggregation

**Quantization**: W8A8 INT8

**Dataset**: 8k+1k

**TPOT**: 50ms

#### Model Deployment

```bash Command
# ============================================================
# Before running, update the following variables:
#   P_IP: prefill node IP address
#   D_IP: decode node IP address
#   ASCEND_MF_STORE_URL: prefill node IP with port
#   MODEL_PATH: path to the model weights directory
#   HCCL_SOCKET_IFNAME: network interface name for HCCL
#   GLOO_SOCKET_IFNAME: network interface name for Gloo
# ============================================================


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
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_OP_EXPANSION_MODE=AIV
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

P_IP=('<your prefill ip>')
D_IP=('<your decode ip>')

export ASCEND_MF_STORE_URL="tcp://<your prefill ip>:24670"

MODEL_PATH=/path/to/model-weights

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"
# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        export DEEPEP_HCCL_BUFFSIZE=2048
        export GLOO_SOCKET_IFNAME=<network-interface>
        export HCCL_SOCKET_IFNAME=<network-interface>
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35
        export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60
        export SGLANG_ENABLE_WAR_BARRIER=1
        export SGLANG_FORCE_COARSE_WAR_BARRIER=1

        python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode prefill \
        --host ${P_IP[$i]} \
        --port 8000 \
        --page-size 128 \
        --tp-size 16 \
        --trust-remote-code \
        --device npu \
        --attention-backend dsv4 \
        --watchdog-timeout 9000 \
        --disaggregation-transfer-backend ascend \
        --disaggregation-bootstrap-port 8998 \
        --mem-fraction-static 0.68 \
        --prefill-max-requests 6 \
        --max-prefill-tokens 80000 \
        --chunked-prefill-size 131072 \
        --max-running-requests 112 \
        --dp-size 16 \
        --enable-dp-attention \
        --moe-a2a-backend deepep \
        --deepep-mode normal \
        --quantization modelslim \
        --enable-dp-lm-head \
        --kv-cache-dtype bfloat16 \
        --disable-cuda-graph \
        --disable-radix-cache \
        --load-balance-method round_robin \
        --ep-dispatch-algorithm static
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
        export GLOO_SOCKET_IFNAME=<network-interface>
        export HCCL_BUFFSIZE=1200
        export HCCL_SOCKET_IFNAME=<network-interface>
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
        export SGLANG_NPU_USE_MULTI_STREAM=1

        python3 -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --disaggregation-mode decode \
        --host ${D_IP[$i]} \
        --port 8001 \
        --page-size 128 \
        --tp-size 16 \
        --trust-remote-code \
        --device npu \
        --attention-backend dsv4 \
        --watchdog-timeout 9000 \
        --mem-fraction-static 0.75 \
        --prefill-max-requests 1 \
        --disable-radix-cache \
        --chunked-prefill-size -1 \
        --disaggregation-transfer-backend ascend \
        --max-running-requests 896 \
        --dp-size 16 \
        --enable-dp-attention \
        --moe-a2a-backend deepep \
        --deepep-mode auto \
        --quantization modelslim \
        --enable-dp-lm-head \
        --kv-cache-dtype bfloat16 \
        --cuda-graph-bs-decode 1 2 4 8 16 24 36 40 48 \
        --speculative-algorithm EAGLE \
        --speculative-num-steps 2 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 3
        break
    fi
done
```

```bash Command
# ============================================================
# Before running, replace the following placeholders:
#   <your prefill ip>: prefill node IP address
#   <your decode ip>: decode node IP address
# ============================================================

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://<your prefill ip>:8000 8998 \
    --decode http://<your decode ip>:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --policy cache_aware
```

#### Benchmark

We tested it based on the `RANDOM` dataset.

```bash Command
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 \
    --port 6688 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 2400 \
    --max-concurrency 800 \
    --random-range-ratio 1 \
    --warmup-requests 16 \
    --request-rate inf \
    --seed 1
```

<a id="single-node-pd-mixed" title="Referenced by external docs. Verify before removing."></a>

### DeepSeek-V4-Flash W8A8 8P IN8K OUT1K 50ms

**Model**: DeepSeek-V4-Flash

**Hardware**: Ascend A3 Series Products

**Cards**: 8

**Deploy Mode**: PD Mixed

**Quantization**: W8A8 INT8

**Dataset**: 8k+1k

**TPOT**: 50ms

#### Model Deployment

```bash Command
# ============================================================
# Before running, update the following variables:
#   MODEL_PATH: path to the model weights directory
#   DRAFT_MODEL_PATH: path to the draft model weights directory
# ============================================================

MODEL_PATH=/path/to/model-weights
DRAFT_MODEL_PATH=/path/to/draft-model-weights

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
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export DEEPEP_HCCL_BUFFSIZE=2500
export DEEPEP_HYBRID_DEPLOYMENT=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ZBAL_BOOTSTRAP_URL=tcp://127.0.0.1:14699
export SGLANG_ZBAL_LOCAL_MEM_SIZE=61000
export STREAMS_PER_DEVICE=32
export ZBAL_ENABLE_GRAPH=1
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 6688 \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --prefill-max-requests 192 \
    --max-prefill-tokens 80000 \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --mem-fraction-static 0.68 \
    --chunked-prefill-size 131072 \
    --max-running-requests 192 \
    --dp-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --quantization modelslim \
    --enable-dp-lm-head \
    --kv-cache-dtype bfloat16 \
    --load-balance-method round_robin \
    --cuda-graph-bs-decode 1 2 4 6 8 10 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 7 \
    --speculative-dspark-block-size 6 \
    --disable-radix-cache
```

#### Benchmark

We tested it based on the `RANDOM` dataset.

```bash Command
python -m sglang.bench_serving \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 \
    --port 6688 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 160 \
    --max-concurrency 160 \
    --random-range-ratio 1 \
    --warmup-requests 0 \
    --request-rate inf \
    --seed 1
```
