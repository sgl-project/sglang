# Best Practice with Qwen Series Modes on Ascend NPU

This section describes the best practice data of mainstream LLM models such as Qwen on the Ascend NPU. If
you encounter issues or have any questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Low-Latency Qwen Series Models

| Model           | Hardware      | Cards | Deploy Mode | Dataset | TPOT | Quantization | Configuration                                                          |
|-----------------|---------------|-------|-------------|---------|------|--------------|------------------------------------------------------------------------|
| Qwen3-235B-A22B | Atlas 800I A3 | 8     | PD Mixed    | 11K+1K  | 10ms | BF16         | [Optimal Configuration](#qwen3-235b-a22b-low-latency-on-a3-mixed-mode) |
| Qwen3-32B       | Atlas 800I A3 | 4     | PD Mixed    | 6K+1.5K | 18ms | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-on-a3-mixed-mode)       |
| Qwen3-32B       | Atlas 800I A3 | 4     | PD Mixed    | 4K+1.5K | 11ms | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-on-a3-mixed-mode)       |
| Qwen3-32B       | Atlas 800I A3 | 8     | PD Mixed    | 18K+4K  | 12ms | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-on-a3-mixed-mode)       |
| Qwen3-32B       | Atlas 800I A2 | 8     | PD Mixed    | 6K+1.5K | 18ms | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-low-latency-on-a2-mixed-mode)       |
| Qwen3-32B       | Atlas 800I A2 | 8     | PD Mixed    | 4K+1.5K | 11ms | BF16         | [Optimal Configuration](#qwen3-32b-low-latency-on-a2-mixed-mode)       |

### Qwen3-235B-A22B Low Latency on A3 Mixed Mode

#### Deployment with 11K+1K TPOT 10ms on 8 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 11K+1K

TPOT: 10ms

Quantization: BF16

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 1 --random-input-len 11000 --random-output-len 1000 --num-prompts 1 --random-range-ratio 1
```

### Qwen3-32B Low Latency on A3 Mixed Mode

#### Deployment with 6K+1.5K TPOT 18ms on 4 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A3 4Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 6K+1.5K

TPOT: 18ms

Quantization: BF16

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32 --random-range-ratio 1
```

#### Deployment with 4K+1.5K TPOT 18ms on 4 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A3 4Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 4K+1.5K

TPOT: 11ms

Quantization: BF16

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --random-range-ratio 1 --max-concurrency 1 --random-output-len 1500 --random-input-len 4096 --num-prompts 4
```

#### Deployment with 18K+4K TPOT 12ms on 8 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 18K+4K

TPOT: 12ms

Quantization: BF16

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7339 --random-range-ratio 1 --max-concurrency 1 --random-output-len 18000 --random-input-len 4000 --num-prompts 1
```

### Qwen3-32B Low Latency on A2 Mixed Mode

#### Deployment with 6K+1.5K TPOT 18ms on 8 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 6K+1.5K

TPOT: 18ms

Quantization: BF16

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32 --random-range-ratio 1
```

#### Deployment with 4K+1.5K TPOT 11ms on 8 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A2 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 4K+1.5K

TPOT: 11ms

Quantization: BF16

- Launch SGLang Instance

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

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 32 \
    --disable-radix-cache \
    --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx  \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 1 4 6 12 18 24 30 32 --dtype bfloat16
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving  --dataset-name random --backend sglang --host 127.0.0.1 --port 7339 --random-range-ratio 1 --max-concurrency 1 --random-output-len 1500 --random-input-len 4096 --num-prompts 4
```

## High-Throughput Qwen Series Models

| Model                          | Hardware      | Cards | Deploy Mode   | Dataset   | TPOT  | Quantization | Configuration                                                                                  |
|--------------------------------|---------------|-------|---------------|-----------|-------|--------------|------------------------------------------------------------------------------------------------|
| Qwen3-235B-A22B                | Atlas 800I A3 | 24    | PD Separation | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-235b-a22b-high-throughput-on-a3-separation-mode)                |
| Qwen3-235B-A22B                | Atlas 800I A3 | 8     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-235b-a22b-high-throughput-on-a3-mixed-mode)                     |
| Qwen3-235B-A22B                | Atlas 800I A3 | 8     | PD Mixed      | 2K+2K     | 100ms | W8A8 INT8    | [Optimal Configuration](#qwen3-235b-a22b-high-throughput-on-a3-mixed-mode)                     |
| Qwen3-235B-A22B                | Atlas 800I A3 | 8     | PD Mixed      | 2K+2K     | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-235b-a22b-high-throughput-on-a3-mixed-mode)                     |
| Qwen3-235B-A22B                | Atlas 800I A3 | 16    | PD Mixed      | 2K+2K     | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-235b-a22b-high-throughput-on-a3-mixed-mode)                     |
| Qwen3-32B                      | Atlas 800I A3 | 2     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-high-throughput-on-a3-mixed-mode)                           |
| Qwen3-32B                      | Atlas 800I A3 | 2     | PD Mixed      | 2K+2K     | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-high-throughput-on-a3-mixed-mode)                           |
| Qwen3-30B-A3B                  | Atlas 800I A3 | 1     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-30b-a3b-high-throughput-on-a3-mixed-mode)                       |
| Qwen3-Coder-480B-A35B-Instruct | Atlas 800I A3 | 24    | PD Separation | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-coder-480b-a35b-instruct-high-throughput-on-a3-separation-mode) |
| Qwen3-Coder-480B-A35B-Instruct | Atlas 800I A3 | 16    | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-coder-480b-a35b-instruct-high-throughput-on-a3-mixed-mode)      |
| Qwen3-Coder-480B-A35B-Instruct | Atlas 800I A3 | 8     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-coder-480b-a35b-instruct-high-throughput-on-a3-mixed-mode)      |
| Qwen3-Next-80B-A3B-Instruct    | Atlas 800I A3 | 2     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-next-80B-a3b-instruct-high-throughput-on-a3-mixed-mode)         |
| Qwen3-32B                      | Atlas 800I A2 | 4     | PD Mixed      | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-high-throughput-on-a2-mixed-mode)                           |
| Qwen3-32B                      | Atlas 800I A2 | 4     | PD Mixed      | 2K+2K     | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-high-throughput-on-a2-mixed-mode)                           |

### Qwen3-235B-A22B High Throughput on A3 Separation Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 24 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 24Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch Prefill and Decode Instance

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

# decode
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

- Launch SGLang Router

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang-oai --host 127.0.0.1 --port 7239 --max-concurrency 860 --random-input-len 3500 --random-output-len 1500 --num-prompts 3440 --random-range-ratio 1
```

### Qwen3-235B-A22B High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 8 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 272 --random-input-len 3500 --random-output-len 1500 --num-prompts 1088 --random-range-ratio 1
```

#### Deployment with 2K+2K TPOT 100ms on 8 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 100ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 576 --random-input-len 2000 --random-output-len 2000 --num-prompts 576 --random-range-ratio 1
```

#### Deployment with 2K+2K TPOT 50ms on 8 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 480 --random-input-len 2048 --random-output-len 2048 --num-prompts 480 --random-range-ratio 1
```

#### Deployment with 2K+2K TPOT 50ms on 16 Cards

- Basic Case Details

Model: Qwen3-235B-A22B-W8A8

Hardware: Atlas 800I A3 16Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 768 --random-input-len 2000 --random-output-len 2000 --num-prompts 768 --random-range-ratio 1
```

### Qwen3-32B High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 2 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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
    --chunked-prefill-size -1 --max-prefill-tokens 49152  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 4  --mem-fraction-static 0.72 --cuda-graph-bs 16 32 64 68 72 78 --dtype bfloat16
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312 --random-range-ratio 1
```

#### Deployment with 2K+2K TPOT 50ms on 2 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 480 --random-range-ratio 1
```

### Qwen3-30B-A3B High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 1 Card

- Basic Case Details

Model: Qwen3-30B-A3B-Instruct-2507

Hardware: Atlas 800I A3 1Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGLang Instance

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
    --max-running-requests 192 \
    --disable-radix-cache \
    --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size -1 --max-prefill-tokens 32768 \
    --tp-size 2 --mem-fraction-static 0.86 --cuda-graph-bs 42 88 96 132 144 156 172 178 192 --dtype bfloat16
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 156 --random-input-len 3500 --random-output-len 1500 --num-prompts 624 --random-range-ratio 1
```


### Qwen3-Coder-480B-A35B-Instruct High Throughput on A3 Separation Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 24 Cards

- Basic Case Details

Model: Qwen3-Coder-480B-A35B-Instruct

Hardware: Atlas 800I A3 24Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch Prefill and Decode Instance

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

- Launch SGLang Router

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 410 --random-input-len 3500 --random-output-len 1500 --num-prompts 1640 --random-range-ratio 1 --request-rate 8
```


### Qwen3-Coder-480B-A35B-Instruct High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 16 Cards

- Basic Case Details

Model: Qwen3-Coder-480B-A35B-Instruct

Hardware: Atlas 800I A3 16Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGlang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 288 --random-input-len 3500 --random-output-len 1500 --num-prompts 1152 --random-range-ratio 1 --request-rate 20
```


#### Deployment with 3.5K+1.5K TPOT 50ms on 8 Cards

- Basic Case Details

Model: Qwen3-Coder-480B-A35B-Instruct

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGlang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 80 --random-input-len 3500 --random-output-len 1500 --num-prompts 320 --random-range-ratio 1
```


### Qwen3-Next-80B-A3B-Instruct High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 2 Cards

- Basic Case Details

Model: Qwen3-Next-80B-A3B-Instruct

Hardware: Atlas 800I A3 2Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGlang Instance

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
        --model-path /path/to/Qwen3-Next-80B-A3B-Instruct-W8A8-3 \
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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 80 --random-output-len 1536 --random-input-len 3584 --num-prompts 160 --random-range-ratio 1
```


### Qwen3-32B High Throughput on A2 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 4 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A2 4Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGlang Instance

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312 --random-range-ratio 1
```


#### Deployment with 2K+2K TPOT 50ms on 4 Cards

- Basic Case Details

Model: Qwen3-32B

Hardware: Atlas 800I A2 4Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W8A8 INT8

- Launch SGlang Instance

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
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \
    --chunked-prefill-size -1 --max-prefill-tokens 49152 --base-gpu-id 4 \
    --tp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs 54 60 66 72 78 84 90 108 114 120 --dtype bfloat16
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 120 --random-range-ratio 1
```
