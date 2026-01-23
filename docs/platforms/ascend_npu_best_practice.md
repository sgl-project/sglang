# Best Practice on Ascend NPU

This section describes the best practice data of mainstream LLM models such as DeepSeek and Qwen on the Ascend Npu. If
you encounter issues or have any questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## DeepSeek Series Models

### Low Latency

| Model         | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | Configuration                                            |
|---------------|---------------|---------|---------------|-----------|--------------|----------------------------------------------------------|
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 6K-1.6K   | W8A8         | [Optimal Configuration](#deepseek-r1-low-latency-20ms-1) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.9K-1K   | W8A8         | [Optimal Configuration](#deepseek-r1-low-latency-20ms-2) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.5K-1.5K | W8A8         | [Optimal Configuration](#deepseek-r1-low-latency-20ms-3) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.5K-1K   | W8A8         | [Optimal Configuration](#deepseek-r1-low-latency-20ms-4) |
| Deepseek-V3.2 | Atlas 800I A3 | 32      | PD Separation | 64K-3K    | W8A8         | [Optimal Configuration](#deepseek-v32-low-latency-30ms)  |

### High Throughput

| Model       | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | Configuration                                                 |
|-------------|---------------|---------|---------------|-----------|--------------|---------------------------------------------------------------|
| Deepseek-R1 | Atlas 800I A3 | 32      | PD Separation | 3.5K-1.5K | W8A8         | [Optimal Configuration](#deepseek-r1-high-performance-50ms-1) |
| Deepseek-R1 | Atlas 800I A3 | 8       | PD Mixed      | 2K-2K     | W4A8         | [Optimal Configuration](#deepseek-r1-high-performance-50ms-2) |
| Deepseek-R1 | Atlas 800I A3 | 16      | PD Separation | 2K-2K     | W4A8         | [Optimal Configuration](#deepseek-r1-high-performance-50ms-3) |
| Deepseek-R1 | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W4A8         | [Optimal Configuration](#deepseek-r1-high-performance-50ms-4) |
| Deepseek-R1 | Atlas 800I A3 | 16      | PD Separation | 3.5K-1.5K | W4A8         | [Optimal Configuration](#deepseek-r1-high-performance-50ms-5) |

## Qwen Series Models

### Low Latency

| Model      | Hardware      | CardNum | Deploy Mode | Dataset | Quantization | Configuration                                           |
|------------|---------------|---------|-------------|---------|--------------|---------------------------------------------------------|
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed    | 11K-1K  | BF16         | [Optimal Configuration](#qwen3-235b-low-latency-10ms)   |
| Qwen3-32B  | Atlas 800I A3 | 4       | PD Mixed    | 6K-1.5K | W8A8         | [Optimal Configuration](#qwen3-32b-low-latency-18ms)    |
| Qwen3-32B  | Atlas 800I A3 | 4       | PD Mixed    | 4K-1.5K | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-11ms)    |
| Qwen3-32B  | Atlas 800I A3 | 8       | PD Mixed    | 18K-4K  | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-12ms)    |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed    | 6K-1.5K | W8A8         | [Optimal Configuration](#qwen3-32b-a2-low-latency-18ms) |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed    | 4K-1.5K | BF16         | [Optimal Configuration](#qwen3-32b-a2-low-latency-11ms) |

### High Throughput

| Model      | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | Configuration                                                 |
|------------|---------------|---------|---------------|-----------|--------------|---------------------------------------------------------------|
| Qwen3-235B | Atlas 800I A3 | 24      | PD Separation | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-1)   |
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-2)   |
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed      | 2K-2K     | W8A8         | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-3)   |
| Qwen3-235B | Atlas 800I A3 | 16      | PD Mixed      | 2K-2K     | W8A8         | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-4)   |
| Qwen3-32B  | Atlas 800I A3 | 2       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-1)    |
| Qwen3-32B  | Atlas 800I A3 | 2       | PD Mixed      | 2K-2K     | W8A8         | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-2)    |
| Qwen3-30B  | Atlas 800I A3 | 1       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-3)    |
| Qwen3-480B | Atlas 800I A3 | 24      | PD Separation | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-1)   |
| Qwen3-480B | Atlas 800I A3 | 16      | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-2)   |
| Qwen3-480B | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-3)   |
| Qwen3-Next | Atlas 800I A3 | 2       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-next-high-throughput-50ms)     |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         | [Optimal Configuration](#qwen3-32b-a2-high-throughput-50ms-1) |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed      | 2K-2K     | W8A8         | [Optimal Configuration](#qwen3-32b-a2-high-throughput-50ms-2) |

## Optimal Configuration

### DeepSeek R1 High Performance 50ms 1

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export ASCEND_MF_STORE_URL="tcp://your prefill ip1:24669"

P_IP=('your prefill ip1' 'your prefill ip2')

D_IP=('your decode ip1' 'your decode ip2')

MODEL_PATH=xxx

export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

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
        export HCCL_BUFFSIZE=1536
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port $((8998+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.81 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 8 --context-length 8192  --disable-radix-cache \
        --chunked-prefill-size -1 --max-prefill-tokens 28680 --moe-a2a-backend deepep --deepep-mode normal \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        --dp-size 2 --enable-dp-attention --disable-shared-experts-fusion --dtype bfloat16 --enable-attn-tp-input-scattered
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=650
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=78
        export TASK_QUEUE_ENABLE=1
        export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 \
        --mem-fraction-static 0.815 --max-running-requests 832 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency --enable-dp-lm-head --moe-dense-tp 1 \
        --cuda-graph-bs 12 14 16 18 20 22 24 26 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3  \
        --tokenizer-worker-num 4 --prefill-round-robin-balance --disable-shared-experts-fusion --dtype bfloat16 \
        --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8998 \
    --prefill http://P_IP:8000 8999 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1 --request-rate 16
```

### DeepSeek R1 Low Latency 20ms 1

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 6K1.6K

TPOT: 20ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_MF_STORE_URL="tcp://your prefill ip1:24669"

P_IP=('your prefill ip1' 'your prefill ip2')

D_IP=('your decode ip1' 'your decode ip2')

MODEL_PATH=xxx

export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

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
        export HCCL_BUFFSIZE=1536
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port $((8998+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.81 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 4 --context-length 8192  --disable-radix-cache \
        --chunked-prefill-size -1 --max-prefill-tokens 28680 --moe-a2a-backend deepep --deepep-mode normal \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        --dp-size 2 --enable-dp-attention --disable-shared-experts-fusion --dtype bfloat16 --enable-attn-tp-input-scattered
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=650
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=12
        export TASK_QUEUE_ENABLE=1
        export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --dist-init-addr DIP1:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 16 \
        --mem-fraction-static 0.75 --max-running-requests 32 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency --enable-dp-lm-head --moe-dense-tp 1 \
        --cuda-graph-bs 2 4 6 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --tokenizer-worker-num 4 --prefill-round-robin-balance --disable-shared-experts-fusion --dtype bfloat16 \
        --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8998 \
    --prefill http://P_IP:8000 8999 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 6000 --random-output-len 1600 --num-prompts 32 --random-range-ratio 1
```

### DeepSeek R1 Low Latency 20ms 2

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 3.9K1K

TPOT: 20ms

#### Model Deployment

Please Turn to [DeepSeek R1 Low Latency 20ms](#deepSeek-r1-low-latency-20ms-1)

#### Benchmark

```bash
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3900 --random-output-len 1000 --num-prompts 768 --random-range-ratio 1 --request-rate 16
```

### DeepSeek R1 Low Latency 20ms 3

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 3.5K1.5K

TPOT: 20ms

#### Model Deployment

Please Turn to [DeepSeek R1 Low Latency 20ms](#deepSeek-r1-low-latency-20ms-1)

#### Benchmark

```bash
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1500 --num-prompts 768 --random-range-ratio 1 --request-rate 16
```

### DeepSeek R1 Low Latency 20ms 4

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 3.5K1K

TPOT: 20ms

#### Model Deployment

Please Turn to [DeepSeek R1 Low Latency 20ms](#deepSeek-r1-low-latency-20ms-1)

#### Benchmark

```bash
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1000 --num-prompts 768 --random-range-ratio 1 --request-rate 16
```

### DeepSeek R1 High Performance 50ms 2

Model: Deepseek R1

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=1600
export DEEPEP_NORMAL_LONG_SEQ_ROUND=10
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

MODEL_PATH=xxx

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6699 \
--cuda-graph-bs 4 8 16 \
--mem-fraction-static 0.74 \
--max-running-requests 256 \
--disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 1500 \
--moe-a2a-backend deepep --deepep-mode auto \
--enable-dp-attention --dp-size 16 --enable-dp-lm-head \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--dtype bfloat16

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 256  --random-input-len 2048 --random-output-len 2048 --num-prompts 1024 --random-range-ratio 1
```

### DeepSeek R1 High Performance 50ms 3

Model: Deepseek R1

Hardware: Atlas 800I A3 16Card

DeployMode: PD Separation

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export ASCEND_MF_STORE_URL="tcp://your prefill ip1:24667"

P_IP=('your prefill ip1')

D_IP=('your decode ip1')

MODEL_PATH=xxx

export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

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
        export HCCL_BUFFSIZE=1536
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port $((8998+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.6 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 8 --context-length 8192  --disable-radix-cache \
        --chunked-prefill-size 32768 --max-prefill-tokens 28680 --moe-a2a-backend deepep --deepep-mode normal \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        --dp-size 2 --enable-dp-attention --disable-shared-experts-fusion --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=720
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
        export TASK_QUEUE_ENABLE=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --nnodes 1 --node-rank 0 --tp-size 16 --dp-size 16 \
        --mem-fraction-static 0.8 --max-running-requests 384 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency --enable-dp-lm-head \
        --cuda-graph-bs 8 10 12 14 16 18 20 22 24 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --prefill-round-robin-balance --disable-shared-experts-fusion --dtype bfloat16 --tokenizer-worker-num 4 \
		    --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8998 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 400  --random-input-len 2048 --random-output-len 2048 --num-prompts 3200 --random-range-ratio 1 --request-rate 8
```

### DeepSeek R1 High Performance 50ms 4

Model: Deepseek R1

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=36
export HCCL_BUFFSIZE=1600
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

MODEL_PATH=xxx

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6699 \
--cuda-graph-bs 8 16 24 28 32 36 \
--mem-fraction-static 0.71 \
--max-running-requests 144 \
--context-length 8188  --disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 9000 \
--moe-a2a-backend deepep --deepep-mode auto \
--enable-dp-attention --dp-size 4 --enable-dp-lm-head \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--dtype bfloat16

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 144  --random-input-len 3500 --random-output-len 1500 --num-prompts 576 --random-range-ratio 1
```

### DeepSeek R1 High Performance 50ms 5

Model: Deepseek R1

Hardware: Atlas 800I A3 16Card

DeployMode: PD Separation

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export ASCEND_MF_STORE_URL="tcp://your prefill ip1:24667"

P_IP=('your prefill ip1')

D_IP=('your decode ip1')

MODEL_PATH=xxx

export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

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
        export HCCL_BUFFSIZE=1536
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 8000 --disaggregation-bootstrap-port $((8998+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.6 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 8 --context-length 8192  --disable-radix-cache \
        --chunked-prefill-size -1 --max-prefill-tokens 28680 --moe-a2a-backend deepep --deepep-mode normal \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        --dp-size 2 --enable-dp-attention --disable-shared-experts-fusion --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export HCCL_BUFFSIZE=720
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
        export TASK_QUEUE_ENABLE=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --nnodes 1 --node-rank 0 --tp-size 16 --dp-size 16 \
        --mem-fraction-static 0.8 --max-running-requests 384 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency --enable-dp-lm-head \
        --cuda-graph-bs 8 10 12 14 16 18 20 22 24 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --prefill-round-robin-balance --disable-shared-experts-fusion --dtype bfloat16 --tokenizer-worker-num 4 \
		    --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://P_IP:8000 8998 \
    --decode http://D_IP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 384  --random-input-len 3500 --random-output-len 1500 --num-prompts 1536 --random-range-ratio 1
```

### Deepseek V32 Low Latency 30ms

Model: Deepseek V3.2

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 64K3K

TPOT: 30ms

#### Model Deployment

Deploy Prefill Instance

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export HCCL_BUFFSIZE=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=5
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

MODEL_PATH=xxx

export SGLANG_NPU_USE_MLAPO=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_OP_EXPANSION_MODE=AIV

IPs=('your prefill ip1' 'your prefill ip2')

# get IP in current node
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo "LOCAL_HOST = " ${LOCAL_HOST}
# get node index
for i in "${!IPs[@]}";
do
  echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
  if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
      echo "Node Rank : ${i}"
      VC_TASK_INDEX=$i
      break
  fi
done

IFNAMES=('xxx' 'xxx')

export HCCL_SOCKET_IFNAME=${IFNAMES[$VC_TASK_INDEX]}
export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
echo "HCCL_SOCKET_IFNAME : ${HCCL_SOCKET_IFNAME}"
nnodes=${#IPs[@]}
tp_size=`expr 16 \* ${nnodes}`
export ASCEND_MF_STORE_URL=tcp://${IPs[0]}:24667

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp $tp_size \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host ${IPs[$VC_TASK_INDEX]} --port 8000 \
--mem-fraction-static 0.73 \
--disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 68000 \
--max-running-requests 1 \
--moe-a2a-backend deepep --deepep-mode normal \
--quantization modelslim \
--disaggregation-transfer-backend ascend \
--disaggregation-mode prefill \
--disable-cuda-graph \
--nnodes $nnodes --node-rank $VC_TASK_INDEX \
--disaggregation-bootstrap-port 8995 \
--enable-nsa-prefill-context-parallel  --moe-dense-tp-size 1 \
--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
--dist-init-addr ${IPs[0]}:10000
```

Deploy Decode Instance

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

MODEL_PATH=xxx

export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_NPU_USE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export TASK_QUEUE_ENABLE=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

IPs=('your decode ip1' 'your decode ip2')

export prefill_ip=your prefill ip1
# get IP in current node
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo "LOCAL_HOST = " ${LOCAL_HOST}
# get node index
for i in "${!IPs[@]}";
do
  echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
  if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
      echo "Node Rank : ${i}"
      VC_TASK_INDEX=$i
      break
  fi
done

IFNAMES=('xxx' 'xxx')

export HCCL_SOCKET_IFNAME=${IFNAMES[$VC_TASK_INDEX]}
export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
nnodes=${#IPs[@]}
tp_size=`expr 16 \* ${nnodes}`
export ASCEND_MF_STORE_URL=tcp://${prefill_ip}:24667

CHUNKED_SIZE=65536
DP=8
export HCCL_BUFFSIZE=400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp $tp_size \
--dp ${DP} \
--ep $tp_size \
--moe-dense-tp-size 1 \
--enable-dp-attention \
--enable-dp-lm-head \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host ${IPs[$VC_TASK_INDEX]} --port 8001 \
--mem-fraction-static 0.79 \
--disable-radix-cache \
--chunked-prefill-size -1 --max-prefill-tokens 68000 \
--max-running-requests 32 \
--cuda-graph-max-bs 4 \
--moe-a2a-backend deepep \
--deepep-mode low_latency \
--quantization modelslim \
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--disaggregation-transfer-backend ascend \
--disaggregation-mode decode \
--prefill-round-robin-balance \
--load-balance-method round_robin \
--nnodes $nnodes --node-rank $VC_TASK_INDEX \
--dist-init-addr ${IPs[0]}:10000 --load-balance-method decode_round_robin
```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://PIP1:8000 8998 \
    --prefill http://PIP2:8000 8999 \
    --decode http://DIP1:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 64000 --random-output-len 3000 --num-prompts 64 --random-range-ratio 1
```

### Qwen3 235B High Throughput 50ms 1

Model: Qwen3 235B

Hardware: Atlas 800I A3 24Card

DeployMode: PD Separation

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

MODEL_PATH=xxx
export ASCEND_MF_STORE_URL="tcp://your prefill ip1:24667"
P_IP=('your prefill ip1')
D_IP=('your decode ip1' 'your decode ip2')
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_DP_ROUND_ROBIN=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
        export HCCL_BUFFSIZE=4300
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export STREAMS_PER_DEVICE=32
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 16 --mem-fraction-static 0.6 \
        --disable-radix-cache \
        --attention-backend ascend --device npu --quantization modelslim --disaggregation-transfer-backend ascend \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --speculative-draft-model-quantization unquant \
        --max-running-requests 128 --chunked-prefill-size 262144 --max-prefill-tokens 262144 \
        --enable-dp-attention  \
        --moe-a2a-backend deepep --deepep-mode normal --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done


for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=24
        export HCCL_BUFFSIZE=512
        export HCCL_SOCKET_IFNAME=data0.3001
        export GLOO_SOCKET_IFNAME=data0.3001
        export STREAMS_PER_DEVICE=32

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 --mem-fraction-static 0.83 --max-running-requests 768 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend ascend_fuseep --cuda-graph-bs 6 8 12 15 18 20 22 24 \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
        --speculative-draft-model-quantization unquant \
        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --dist-init-addr xxx:5000 \
        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --prefill-round-robin-balance --enable-dp-lm-head --dtype bfloat16 --tokenizer-worker-num 4 \
        --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://PIP:8000 8995 \
    --decode http://DIP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang-oai --host 127.0.0.1 --port 7239 --max-concurrency 860 --random-input-len 3500 --random-output-len 1500 --num-prompts 3440 --random-range-ratio 1
```

### Qwen3 235B High Throughput 50ms 2

Model: Qwen3 235B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=2

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu --quantization modelslim  \
    --max-running-requests 272 --context-length 8192 --dtype bfloat16 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --disable-radix-cache --moe-a2a-backend deepep  --deepep-mode auto --speculative-draft-model-quantization unquant \
    --tp 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --mem-fraction-static 0.8 --cuda-graph-bs 3 4 6 8 10 12 13 14 15 16 17

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 272 --random-input-len 3500 --random-output-len 1500 --num-prompts 1088 --random-range-ratio 1
```

### Qwen3-235B Atlas 800I A3-8Card PD Mixed 2K-2K 100ms

Model: Qwen3 235B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 100ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1200
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu --quantization modelslim  \
    --max-running-requests 576 --context-length 8192 --dtype bfloat16 \
    --chunked-prefill-size 32768 --max-prefill-tokens 458880  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --disable-radix-cache --moe-a2a-backend deepep  --deepep-mode auto --speculative-draft-model-quantization unquant  \
    --tp 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --mem-fraction-static 0.81 --cuda-graph-bs 8 16 20 24 32 36

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 576 --random-input-len 2000 --random-output-len 2000 --num-prompts 576 --random-range-ratio 1
```

### Qwen3 235B High Throughput 50ms 3

Model: Qwen3 235B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=2100
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu --quantization modelslim  \
    --max-running-requests 480 --context-length 8192 --dtype bfloat16 \
    --chunked-prefill-size -1 --max-prefill-tokens 4096 --speculative-draft-model-quantization unquant  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --disable-radix-cache --moe-a2a-backend deepep  --deepep-mode auto  \
    --tp 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --mem-fraction-static 0.75 --cuda-graph-bs 6 8 10 12 15 18 28 30
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 480 --random-input-len 2048 --random-output-len 2048 --num-prompts 480 --random-range-ratio 1
```

### Qwen3 235B High Throughput 50ms 4

Model: Qwen3 235B

Hardware: Atlas 800I A3 16Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"

MIX_IP=('IP1' 'IP2')

for i in "${!MIX_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${MIX_IP[$i]}" || "$LOCAL_HOST2" == "${MIX_IP[$i]}" ]];
    then
        echo "${MIX_IP[$i]}"
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} \
        --host 127.0.0.1 --port 7439 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 --mem-fraction-static 0.8 --max-running-requests 768 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend deepep --deepep-mode auto --cuda-graph-bs 6 8 10 12 18 24 \
        --dist-init-addr ${MIX_IP[0]}:5000 --chunked-prefill-size 131072 --max-prefill-tokens 458880 \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx --speculative-draft-model-quantization= unquant \
        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --context-length 8192 --disable-radix-cache \
        --enable-dp-lm-head --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 768 --random-input-len 2000 --random-output-len 2000 --num-prompts 768 --random-range-ratio 1
```

### Qwen3 235B Low Latency 10ms

Model: Qwen3 235B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 11K1K

TPOT: 10ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu --quantization modelslim  \
    --max-running-requests 1  --dtype bfloat16 \
    --chunked-prefill-size -1 --max-prefill-tokens 16384 --speculative-draft-model-quantization unquant  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --disable-radix-cache --enable-dp-lm-head \
    --tp 16 --mem-fraction-static 0.78 --cuda-graph-bs 1

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 1 --random-input-len 11000 --random-output-len 1000 --num-prompts 1 --random-range-ratio 1
```

### Qwen3 32B Low Latency 18ms

Model: Qwen3 32B

Hardware: Atlas 800I A3 4Card

DeployMode: PD Mixed

DataSets: 6K1.5K

TPOT: 18ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu \
    --max-running-requests 32 \
    --disable-radix-cache \
    --chunked-prefill-size 24576 --max-prefill-tokens 65536 \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 8 16 24 32  --dtype bfloat16

```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32 --random-range-ratio 1
```

### Qwen3 32B Low Latency 11ms

Model: Qwen3 32B

Hardware: Atlas 800I A3 4Card

DeployMode: PD Mixed

DataSets: 4K1.5K

TPOT: 11ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size 24576 --max-prefill-tokens 65536  \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 1 --dtype bfloat16

```

#### Benchmark

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --random-range-ratio 1 --max-concurrency 1 --random-output-len 1500 --random-input-len 4096 --num-prompts 4
```

### Qwen3 32B Low Latency 12ms

Model: Qwen3 32B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 18K4K

TPOT: 12ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 1 \
    --disable-radix-cache --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --tp-size 16 --mem-fraction-static 0.72 --cuda-graph-bs 1 --dtype bfloat16
```

#### Benchmark

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7339 --random-range-ratio 1 --max-concurrency 1 --random-output-len 18000 --random-input-len 4000 --num-prompts 1
```

### Qwen3 32B High Throughput 50ms 1

Model: Qwen3 32B

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH


MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 78 \
    --disable-radix-cache --speculative-draft-model-quantization unquant \
    --chunked-prefill-size 65536 --max-prefill-tokens 65536  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 4  --mem-fraction-static 0.7 --cuda-graph-bs 16 32 64 68 72 78 --dtype bfloat16
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312 --random-range-ratio 1
```

### Qwen3 32B High Throughput 50ms 2

Model: Qwen3 32B

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 120 \
    --disable-radix-cache --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size -1 --max-prefill-tokens 49152 \
    --tp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs 54 60 66 72 78 84 90 108 114 120 --dtype bfloat16

```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 480 --random-range-ratio 1
```

### Qwen3 32B High Throughput 50ms 3

Model: Qwen3 30B

Hardware: Atlas 800I A3 1Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 192 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size -1 --max-prefill-tokens 32768 \
    --tp-size 2 --mem-fraction-static 0.86 --cuda-graph-bs 42 88 96 132 144 156 172 178 192 --dtype bfloat16
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 156 --random-input-len 3500 --random-output-len 1500 --num-prompts 624 --random-range-ratio 1
```

### Qwen3 480B High Throughput 50ms 1

Model: Qwen3 480B

Hardware: Atlas 800I A3 24Card

DeployMode: PD Separation

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

MODEL_PATH=xxx
export ASCEND_MF_STORE_URL="tcp://PIP:24667"
P_IP=('PIP')
D_IP=('DIP1' 'DIP2')
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
        export HCCL_BUFFSIZE=4300
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export STREAMS_PER_DEVICE=32
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 2 --mem-fraction-static 0.6 \
        --disable-radix-cache \
	      --attention-backend ascend --device npu --quantization modelslim --disaggregation-transfer-backend ascend \
	      --max-running-requests 128 --chunked-prefill-size 65536 --max-prefill-tokens 262144 \
        --enable-dp-attention  \
        --moe-a2a-backend deepep --deepep-mode normal --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=72
        export HCCL_BUFFSIZE=512
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        export STREAMS_PER_DEVICE=32

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 4 --mem-fraction-static 0.73 --max-running-requests 384 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend ascend_fuseep --cuda-graph-bs 16 32 48 56 64 72 80 88 96 \
        --dist-init-addr DIP1:5000 \
	      --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --prefill-round-robin-balance --enable-dp-lm-head --dtype bfloat16 --tokenizer-worker-num 4 --load-balance-method decode_round_robin
        NODE_RANK=$i
        break
    fi
done

```

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://PIP:8000 8995 \
    --decode http://DIP:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 410 --random-input-len 3500 --random-output-len 1500 --num-prompts 1640 --random-range-ratio 1 --request-rate 8
```

### Qwen3 480B High Throughput 50ms 2

Model: Qwen3 480B

Hardware: Atlas 800I A3 16Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=1800
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"

MIX_IP=('IP1' 'IP2')

for i in "${!MIX_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${MIX_IP[$i]}" || "$LOCAL_HOST2" == "${MIX_IP[$i]}" ]];
    then
        echo "${MIX_IP[$i]}"

        python -m sglang.launch_server --model-path $MODEL_PATH \
        --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 2 --node-rank $i  \
        --dist-init-addr 141.61.133.128:5000 \
        --attention-backend ascend --device npu --quantization modelslim  \
        --max-running-requests 288 --context-length 8192 --dtype bfloat16  \
        --chunked-prefill-size 114688 --max-prefill-tokens 458880  \
        --disable-radix-cache --moe-a2a-backend deepep  --deepep-mode auto  \
        --tp 32 --dp-size 4 --enable-dp-attention --enable-dp-lm-head --mem-fraction-static 0.7 --cuda-graph-bs 56 64 72
        NODE_RANK=$i
        break
    fi
done
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 288 --random-input-len 3500 --random-output-len 1500 --num-prompts 1152 --random-range-ratio 1 --request-rate 20
```

### Qwen3 480B High Throughput 50ms 3

Model: Qwen3 480B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=2100
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"

python -m sglang.launch_server --model-path $MODEL_PATH \
--host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
--attention-backend ascend --device npu --quantization modelslim  \
--max-running-requests 80 --context-length 8192 --dtype bfloat16 \
--chunked-prefill-size 28672 --max-prefill-tokens 458880  \
--disable-radix-cache --moe-a2a-backend deepep  --deepep-mode auto --enable-dp-attention --enable-dp-lm-head \
--tp 16 --dp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs  16 20 24
```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 80 --random-input-len 3500 --random-output-len 1500 --num-prompts 320 --random-range-ratio 1
```

### Qwen3 Next High Throughput 50ms

Model: Qwen3 Next

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
export cann_path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/driver/bin/setenv.bash
source ${cann_path}/../set_env.sh
source ${cann_path}/../../nnal/atb/set_env.sh
source ${cann_path}/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=${cann_path}
source /usr/local/Ascend/8.5.0/bisheng_toolkit/set_env.sh

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_ALGO="level0:NA;level1:ring"

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=20
export HCCL_BUFFSIZE=2000

python -m sglang.launch_server \
        --model-path /mnt/share/weight/Qwen3-Next-80B-A3B-Instruct-W8A8-3 \
        --host 127.0.0.1 \
        --port 6699 \
        --tp-size 4 \
        --device npu \
        --attention-backend ascend \
        --mem-fraction-static 0.685 \
        --max-running-requests 80 \
        --watchdog-timeout 3600 \
        --disable-radix-cache \
        --cuda-graph-bs 80 \
        --max-prefill-tokens 28672  --max-total-tokens 450560 \
        --moe-a2a-backend deepep --deepep-mode auto \
        --quantization modelslim \
        --chunked-prefill-size -1
```

#### Benchmark

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 80 --random-output-len 1536 --random-input-len 3584 --num-prompts 160 --random-range-ratio 1
```

### Qwen3 32B A2 Low Latency 18ms

Model: Qwen3 32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

DataSets: 6K1.5K

TPOT: 18ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7439 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 32 \
    --disable-radix-cache \
    --chunked-prefill-size 24576 --max-prefill-tokens 65536 \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 8 16 24 32 --dtype bfloat16
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32 --random-range-ratio 1
```

### Qwen3 32B A2 Low Latency 11ms

Model: Qwen3 32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

DataSets: 4K1.5K

TPOT: 11ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 32 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx  \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 1 4 6 12 18 24 30 32 --dtype bfloat16
```

#### Benchmark

```shell
python3 -m sglang.bench_serving  --dataset-name random --backend sglang --host 127.0.0.1 --port 7339 --random-range-ratio 1 --max-concurrency 1 --random-output-len 1500 --random-input-len 4096 --num-prompts 4
```

### Qwen3 32B A2 High Throughput 50ms 1

Model: Qwen3 32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 78 \
    --disable-radix-cache --speculative-draft-model-quantization unquant \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 4  --mem-fraction-static 0.72 --cuda-graph-bs 1 4 8 16 32 64 68 72 78 --dtype bfloat16 --base-gpu-id 4
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312 --random-range-ratio 1
```

### Qwen3 32B A2 High Throughput 50ms 2

Model: Qwen3 32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

DataSets: 2K2K

TPOT: 50ms

#### Model Deployment

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

MODEL_PATH=xxx

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export HCCL_BUFFSIZE=400
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 120 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \
    --chunked-prefill-size -1 --max-prefill-tokens 49152 --base-gpu-id 4 \
    --tp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs 54 60 66 72 78 84 90 108 114 120 --dtype bfloat16
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 120 --random-range-ratio 1
```
