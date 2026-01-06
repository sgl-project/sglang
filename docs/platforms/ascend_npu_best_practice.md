# Best Practice on Ascend NPU

This section describes the best practice data of mainstream LLM models such as DeepSeek and Qwen on the Ascend Npu.If
you encounter issues or have any questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## DeepSeek Series Models

### Low Latency

| Model         | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | TPOT(ms) | Output TPS(per card) | Configuration                                            |
|---------------|---------------|---------|---------------|-----------|--------------|:--------:|:--------------------:|----------------------------------------------------------|
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 6K-1.6K   | W8A8         |  19.81   |        36.906        | [Optimal Configuration](#deepSeek-r1-low-latency-20ms-1) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.9K-1K   | W8A8         |  19.77   |        35.625        | [Optimal Configuration](#deepSeek-r1-low-latency-20ms-2) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.5K-1.5K | W8A8         |  19.92   |        36.980        | [Optimal Configuration](#deepSeek-r1-low-latency-20ms-3) |
| Deepseek-R1   | Atlas 800I A3 | 32      | PD Separation | 3.5K-1K   | W8A8         |  19.52   |        36.344        | [Optimal Configuration](#deepSeek-r1-low-latency-20ms-4) |
| Deepseek-V3.2 | Atlas 800I A3 | 32      | PD Separation | 64K-1K    | W8A8         |  25.36   |        14.74         | [Optimal Configuration](#deepseek-v32-low-latency-30ms)  |

### High Throughput

| Model       | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | TPOT(ms) | Output TPS(per card) | Configuration                                                 |
|-------------|---------------|---------|---------------|-----------|--------------|:--------:|:--------------------:|---------------------------------------------------------------|
| Deepseek-R1 | Atlas 800I A3 | 32      | PD Separation | 3.5K-1.5K | W8A8         |  48.10   |       396.796        | [Optimal Configuration](#deepSeek-r1-high-performance-50ms-1) |
| Deepseek-R1 | Atlas 800I A3 | 8       | PD Mixed      | 2K-2K     | W4A8         |  49.67   |       528.375        | [Optimal Configuration](#deepSeek-r1-high-performance-50ms-2) |
| Deepseek-R1 | Atlas 800I A3 | 16      | PD Separation | 2K-2K     | W4A8         |  47.76   |       452.227        | [Optimal Configuration](#deepSeek-r1-high-performance-50ms-3) |
| Deepseek-R1 | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W4A8         |  49.77   |       312.077        | [Optimal Configuration](#deepSeek-r1-high-performance-50ms-4) |
| Deepseek-R1 | Atlas 800I A3 | 16      | PD Separation | 3.5K-1.5K | W4A8         |  49.43   |       361.500        | [Optimal Configuration](#deepSeek-r1-high-performance-50ms-5) |

## Qwen Series Models

### Low Latency

| Model      | Hardware      | CardNum | Deploy Mode | Dataset | Quantization | TPOT(ms) | Output TPS(per card) | Configuration                                           |
|------------|---------------|---------|-------------|---------|--------------|:--------:|:--------------------:|---------------------------------------------------------|
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed    | 11K-1K  | BF16         |   9.70   |        11.690        | [Optimal Configuration](#qwen3-235b-low-latency-10ms)   |
| Qwen3-32B  | Atlas 800I A3 | 4       | PD Mixed    | 6K-1.5K | W8A8         |  16.87   |       311.750        | [Optimal Configuration](#qwen3-32b-low-latency-18ms)    |
| Qwen3-32B  | Atlas 800I A3 | 4       | PD Mixed    | 4K-1.5K | BF16         |   9.46   |        25.850        | [Optimal Configuration](#qwen3-32b-low-latency-11ms)    |
| Qwen3-32B  | Atlas 800I A3 | 8       | PD Mixed    | 18K-4K  | BF16         |  12.27   |        9.955         | [Optimal Configuration](#qwen3-32b-low-latency-12ms)    |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed    | 6K-1.5K | W8A8         |  16.46   |         296          | [Optimal Configuration](#qwen3-32b-a2-low-latency-18ms) |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed    | 4K-1.5K | BF16         |  10.18   |          12          | [Optimal Configuration](#qwen3-32b-a2-low-latency-11ms) |

### High Throughput

| Model      | Hardware      | CardNum | Deploy Mode   | Dataset   | Quantization | TPOT(ms) | Output TPS(per card) | Configuration                                                 |
|------------|---------------|---------|---------------|-----------|--------------|:--------:|:--------------------:|---------------------------------------------------------------|
| Qwen3-235B | Atlas 800I A3 | 24      | PD Separation | 3.5K-1.5K | W8A8         |  40.75   |       467.416        | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-1)   |
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         |  51.51   |       477.625        | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-2)   |
| Qwen3-235B | Atlas 800I A3 | 8       | PD Mixed      | 2K-2K     | W8A8         |  54.78   |       790.071        | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-3)   |
| Qwen3-235B | Atlas 800I A3 | 16      | PD Mixed      | 2K-2K     | W8A8         |  50.12   |       519.625        | [Optimal Configuration](#qwen3-235b-high-throughput-50ms-4)   |
| Qwen3-32B  | Atlas 800I A3 | 2       | PD Mixed      | 3.5K-1.5K | W8A8         |  49.20   |       707.500        | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-1)    |
| Qwen3-32B  | Atlas 800I A3 | 2       | PD Mixed      | 2K-2K     | W8A8         |  48.30   |       986.150        | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-2)    |
| Qwen3-30B  | Atlas 800I A3 | 1       | PD Mixed      | 3.5K-1.5K | W8A8         |  44.35   |       3166.030       | [Optimal Configuration](#qwen3-32b-high-throughput-50ms-3)    |
| Qwen3-480B | Atlas 800I A3 | 24      | PD Separation | 3.5K-1.5K | W8A8         |  48.27   |       266.250        | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-1)   |
| Qwen3-480B | Atlas 800I A3 | 16      | PD Mixed      | 3.5K-1.5K | W8A8         |  50.34   |       289.813        | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-2)   |
| Qwen3-480B | Atlas 800I A3 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         |  48.20   |       187.500        | [Optimal Configuration](#qwen3-480b-high-throughput-50ms-3)   |
| Qwen3-Next | Atlas 800I A3 | 2       | PD Mixed      | 3.5K-1.5K | W8A8         |  49.91   |        702.83        | [Optimal Configuration](#qwen3-next-high-throughput-50ms)     |                                                         |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed      | 3.5K-1.5K | W8A8         |  48.97   |        348.75        | [Optimal Configuration](#qwen3-32b-a2-high-throughput-50ms-1) |
| Qwen3-32B  | Atlas 800I A2 | 8       | PD Mixed      | 2K-2K     | W8A8         |  45.88   |         512          | [Optimal Configuration](#qwen3-32b-a2-high-throughput-50ms-2) |

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

D_IP=('your prefill ip1' 'your prefill ip2')

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
        export TASK_QUEUE_ENABLE=0
        export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --dist-init-addr DIP1:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 \
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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 832  --random-input-len 3500 --random-output-len 1500 --num-prompts 3328 --random-range-ratio 1
```

```angular2html
============ Serving Benchmark Result ============
Backend: sqlang
Traffic request rate: inf
Max request concurrency: 832
Successful requests: 3328
Benchmark duration (s): 393.15
Total input tokens: 11202837
Total input text tokens: 11202837
Total input vision tokens: 0
Total generated tokens (retokenized): 4992000
Request throughput (req/s): 8.465
Output token throughput (tok/s): 12697.4868
Total token throughput (tok/s): 41192.654
Concurrency: 696.859
-----------------End-to-End Latency----------------
Mean E2E Latency (ms): 82322.96
Median E2E Latency (ms): 82395.75
----------------Time to First Token----------------
Mean TTFT (ms): 10170.34
Median TTFT (ms): 8273.99
P99 TTFT (ms): 29787.93
----------------Output Token (excl. 1st token)----------------
Mean PTOT (ms): 48.10
Median PTOT (ms): 49.12
P99 PTOT (ms): 55.02
----------------Inter-Token Latency----------------
Mean ITL (ms): 122.53
Median ITL (ms): 120.08
P99 ITL (ms): 278.32
Max ITL (ms): 838.11
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
        export TASK_QUEUE_ENABLE=0
        export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_SOCKET_IFNAME=xxx
        export GLOO_SOCKET_IFNAME=xxx
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode --host ${D_IP[$i]} \
        --port 8001 --trust-remote-code --dist-init-addr DIP1:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 16 \
        --mem-fraction-static 0.75 --max-running-requests 32 --attention-backend ascend --device npu --quantization modelslim \
        --moe-a2a-backend deepep --enable-dp-attention --deepep-mode low_latency --enable-dp-lm-head --moe-dense-tp 1 \
        --cuda-graph-bs 4 --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
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

```angular2html
========== Serving Benchmark Result ==========
Backend:                             sglang
Traffic request rate:                inf
Max request concurrency:             32
Successful requests:                 32
Benchmark duration (s):              43.34
Total input tokens:                  192000
Total input text tokens:             192000
Total input vision tokens:           0
Total generated tokens:              51200
Total generated tokens (retokenized): 50985
Request throughput (req/s):          0.74
Input token throughput (tok/s):      4429.93
Output token throughput (tok/s):     1181.31
Peak output token throughput (tok/s): 1667.00
Peak concurrent requests:            32
Total token throughput (tok/s):      5611.24
Concurrency:                         27.19
Accept length:                       2.59
---------- End-To-End Latency ----------
Mean E2E Latency (ms):               36921.96
Median E2E Latency (ms):             36785.75
---------- Time to First Token ----------
Mean TTFT (ms):                      5149.34
Median TTFT (ms):                    5146.99
P99 TTFT (ms):                       9015.93
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                      19.81
Median TPOT (ms):                    20.22
P99 TPOT (ms):                       23.74
---------- Inter-Token Latency ----------
Mean ITL (ms):                       19.81
Median ITL (ms):                     19.08
P95 ITL (ms):                        29.95
P99 ITL (ms):                        55.85
Max ITL (ms):                        123.03
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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 3900 --random-output-len 1000 --num-prompts 32 --random-range-ratio 1
```

```angular2html
========== Serving Benchmark Result ==========
Backend:                              sglang
Traffic request rate:                 inf
Max request concurrency:              32
Successful requests:                  32
Benchmark duration (s):               28.07
Total input tokens:                   124800
Total input text tokens:              0
Total generated vision tokens:        32000
Total generated tokens (retokenized): 31809
Request throughput (req/s):           1.14
Input token throughput (tok/s):       4446.13
Output token throughput (tok/s):      1148.03
Peak output token throughput (tok/s): 1702.00
Peak concurrent requests:             32
Output token throughput (tok/s):      5586.16
Concurrency:                          25.82
Accept length:                        2.88
---------- End-to-End Latency ----------
Mean E2E Latency (ms):                22650.75
Median E2E Latency (ms):              22649.04
---------- Time to First Token ----------
Mean TTFT (ms):                       2901.77
Median TTFT (ms):                     2357.79
P99 TTFT (ms):                        4240.04
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                       19.77
Median TPOT (ms):                     19.68
P99 TPOT (ms):                        23.90
---------- Inter-Token Latency ----------
Mean ITL (ms):                        19.77
Median ITL (ms):                      19.93
P95 ITL (ms):                         29.72
P99 ITL (ms):                         57.10
Max ITL (ms):                         122.71
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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 3500 --random-output-len 1500 --num-prompts 32 --random-range-ratio 1
```

```angular2html
========== Serving Benchmark Result ==========
Backend:                              sglang
Traffic request rate:                 inf
Max request concurrency:              32
Successful requests:                  32
Benchmark duration (s):               40.56
Total input tokens:                   112000
Total input text tokens:              112000
Total input vision tokens:            0
Total generated tokens:               48000
Total generated tokens (retokenized): 47787
Request throughput (req/s):           0.79
Input token throughput (tok/s):       2761.16
Output token throughput (tok/s):      1183.35
Peak output token throughput (tok/s): 1665.00
Peak concurrent requests:             32
Total token throughput (tok/s):       3944.51
Concurrency:                          25.66
Accept length:                        2.84
---------- End-to-End Latency ----------
Mean E2E Latency (ms):                32526.35
Median E2E Latency (ms):              32200.67
---------- Time to First Token ----------
Mean TTFT (ms):                       2671.91
Median TTFT (ms):                     2208.48
P99 TTFT (ms):                        3960.99
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                       19.92
Median TPOT (ms):                     19.91
P99 TPOT (ms):                        24.64
---------- Inter-Token Latency ----------
Mean ITL (ms):                        19.92
Median ITL (ms):                      18.91
P95 ITL (ms):                         30.06
P99 ITL (ms):                         57.76
Max ITL (ms):                         189.36
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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 3500 --random-output-len 1000 --num-prompts 32 --random-range-ratio 1
```

```angular2html
Backend:
================ Serving Benchmark Result ===================
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     32
Benchmark duration (s):                  27.50
Total input tokens:                      112000
Total input text tokens:                 0
Total input vision tokens:               0
Total generated tokens:                  32000
Total generated tokens (retokenized):    31837
Request throughput (req/s):              1.16
Input token throughput (tok/s):          4072.55
Output token throughput (tok/s):         1163.59
Peak output token throughput (tok/s):    1692.00
Peak concurrent requests:                32
Total token throughput (tok/s):          5236.14
Concurrency:                             25.78
Accept length:                           2.84
------------------ End-To-End Latency ------------------
Mean E2E Latency (ms):                   22154.99
Median E2E Latency (ms):                 22262.70
------------------ Time to First Token ------------------
Mean TTFT (ms):                          2655.44
Median TTFT (ms):                        2205.10
P99 TTFT (ms):                           4446.79
------------------ Time per Output Token (excl. 1st token) ------------------
Mean TPOT (ms):                          19.52
Median TPOT (ms):                        19.82
P99 TPOT (ms):                           25.19
------------------ Inter-Token Latency ------------------
Mean ITL (ms):                           19.52
Median ITL (ms):                         18.81
P99 ITL (ms):                            29.56
Max ITL (ms):                            65.46
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

```angular2html
============= Serving Benchmark Result ================
Backend                                 sglang
Traffic request rate:                   inf
Max request concurrency:                255
Successful requests:                    1024
Benchmark duration (s):                 496.11
Total input tokens:                     2097152
Total input text tokens:                2097152
Total input vision tokens:              0
Total generated tokens:                 2097152
Total generated tokens (retokenized):   2091432
Request throughput (req/s):             2.06
Input token throughput (tok/s):         4227.20
Output token throughput (tok/s):        4227.20
Peak output token throughput (tok/s):   7098.00
Peak concurrent requests:               275
Total token throughput (tok/s):         8454.39
Concurrency:                            231.08
Accept length:                          2.82
====================End-To-End Latency============
Mean E2E Latency (ms):                  111956.34
Median E2E Latency (ms):                113251.95
====================Time to First Token===========
Mean TTFT (ms):                         10273.92
Median TTFT (ms):                       8592.03
P99 TTFT (ms):                          28535.07
=== Time per Output Token (excl. 1st tok) ===
Mean TPOT (ms):                         49.67
Median TPOT (ms):                       50.30
P99 TPOT (ms):                          66.12
====================Inter-Token Latency===========
Mean ITL (ms):                          49.67
Median ITL (ms):                        34.83
P99 ITL (ms):                           75.91
Max ITL (ms):                           18621.17
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
        export TASK_QUEUE_ENABLE=0
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

```angular2html
==================== Serving Benchmark Result ====================
Backend:                             sglang
Traffic request rate:                8.0
Max request concurrency:             400
Successful requests:                 3200
Benchmark duration (s):              905.74
Total input tokens:                  6553600
Total input text tokens:             6553600
Total input vision tokens:           0
Total generated tokens (retokenized): 6534368
Request throughput (req/s):          3.53
Input token throughput (tok/s):      7235.64
Output token throughput (tok/s):     7235.64
Peak output token throughput (tok/s): 9112.00
Peak concurrent requests:            411
Total token throughput (tok/s):      14471.28
Concurrency:                         363.54
Accept length:                       2.86
------------------------ End-to-End Latency ------------------------
Mean E2E Latency (ms):               102896.35
Median E2E Latency (ms):             104894.80
------------------------ Time to First Token -----------------------
Mean TTFT (ms):                      5138.54
Median TTFT (ms):                    3356.93
P99 TTFT (ms):                       20223.25
------------------ Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                      47.76
Median TPOT (ms):                    48.84
P99 TPOT (ms):                       62.18
------------------------ Inter-Token Latency -----------------------
Mean ITL (ms):                       47.76
Median ITL (ms):                     40.66
P95 ITL (ms):                        83.83
P99 ITL (ms):                        147.10
Max ITL (ms):                        674.12
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

```angular2html
--------------- Serving Benchmark Result -----------------
Backend:                             sglang
Traffic request rate:                inf
Max request concurrency:             144
Successful requests:                 576
Benchmark duration (s):              346.07
Total input tokens:                  2016000
Total input text tokens:             2016000
Total input vision tokens:           0
Total generated tokens:              864000
Total generated tokens (retokenized): 861521
Request throughput (req/s):          1.66
Input token throughput (tok/s):      5825.46
Output token throughput (tok/s):     2496.62
Peak output token throughput (tok/s): 4724.00
Peak concurrent requests:            157
Total token throughput (tok/s):      8322.08
Concurrency:                         135.48
Accept length:                       2.83
-----------------------------------------------------------
-------- End-to-End Latency --------
Mean E2E Latency (ms):               81395.45
Median E2E Latency (ms):             82243.44
-------- Time to First Token --------
Mean TTFT (ms):                      6788.93
Median TTFT (ms):                    3911.21
P99 TTFT (ms):                       26898.02
-------- Time per Output Token (excl. 1st token) --------
Mean TPOT (ms):                      49.77
Median TPOT (ms):                    50.36
P99 TPOT (ms):                       68.57
-------- Inter-Token Latency --------
Mean ITL (ms):                       49.77
Median ITL (ms):                     29.30
P95 ITL (ms):                        92.28
P99 ITL (ms):                        509.53
Max ITL (ms):                        19821.92
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
        export TASK_QUEUE_ENABLE=0
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

```angular2html
=======================================
Serving Benchmark Result
=======================================
Backend:                             sglang
Traffic request rate:                8.0
Max request concurrency:             384
Successful requests:                 1536
Benchmark duration (s):              398.28
Total input tokens:                  5376000
Total input text tokens:             5376000
Total input vision tokens:           0
Total generated tokens (retokenized): 2297661
Request throughput (req/s):          3.86
Input token throughput (tok/s):      13498.06
Output token throughput (tok/s):     5784.88
Peak output token throughput (tok/s): 8359.00
Peak concurrent requests:            395
Total token throughput (tok/s):      19282.95
Concurrency:                         323.46
Accept length:                       2.87
---------------------------------------
End-to-End Latency
---------------------------------------
Mean E2E Latency (ms):               83871.76
Median E2E Latency (ms):             84454.65
---------------------------------------
Time to First Token
---------------------------------------
Mean TTFT (ms):                      9784.06
Median TTFT (ms):                    6542.10
P99 TTFT (ms):                       31487.26
---------------------------------------
Time per Output Token (excl. 1st token)
---------------------------------------
Mean TPOT (ms):                      49.42
Median TPOT (ms):                    50.13
P99 TPOT (ms):                       66.06
---------------------------------------
Inter-Token Latency
---------------------------------------
Mean ITL (ms):                       49.43
Median ITL (ms):                     41.87
P99 ITL (ms):                        85.42
Max ITL (ms):                        672.12
```

### Deepseek V32 Low Latency 30ms

Model: Deepseek V3.2

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

DataSets: 64K1K

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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 64000 --random-output-len 1000 --num-prompts 64 --random-range-ratio 1 --request-rate 0.25
```

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: 0.25
Max request concurrency: 32
Successful requests: 64
Benchmark duration (s): 416.83
Total input tokens: 4096000
Total input text tokens: 4096000
Total generated tokens (retokenized): 196608
Total generated tokens (retokenized): 196530
Request throughput (req/s): 0.15
Input token throughput (tok/s): 98866.58
Output token throughput (tok/s): 471.68
Peak output token throughput (tok/s): 821.00
Peak concurrent requests: 33
Total token throughput (tok/s): 10298.25
Concurrency: 17.99
Accept length: 3.18
----------------- End-to-End Latency -----------------
Mean E2E Latency (ms): 117181.11
Median E2E Latency (ms): 100840.21
Mean TTFT (ms): 39313.10
Median TTFT (ms): 19198.87
P99 TTFT (ms): 153478.05
Mean Time per Output Token (excl. 1st token) (ms): 25.25
Mean TPOT (ms): 25.52
P99 TPOT (ms): 29.63
----------------- Inter-Token Latency -----------------
Mean ITL (ms): 25.36
Median ITL (ms): 20.47
P95 ITL (ms): 40.62
P99 ITL (ms): 54.63
Max ITL (ms): 248.81
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
        --ep-dispatch-algorithm static --init-expert-location /mnt/share/chenxu/hot_map/expert_distribution_recorder_1765615213.9892833.pt \
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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 768 --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1
```

```angular2htm
========= Serving Benchmark Result =========
Backend:                      sglang
Traffic request rate:         inf
Max request concurrency:      768
Successful requests:          3072
Benchmark duration (s):       410.76
Total input tokens:           10752000
Total input text tokens:      10752000
Total generated tokens (retokenized):  46080000
Request throughput (req/s):   7.48
Input token throughput (tok/s): 26176.11
Output token throughput (tok/s): 11218.33
Peak output token throughput (tok/s): 15636.00
Peak concurrent requests:     787
Total token throughput (tok/s): 37394.45
Concurrency:                  658.11
Accept length:                2.87
----------------- End-to-End Latency -----------------
Mean E2E Latency (ms):        87995.10
Median E2E Latency (ms):      80821.00
Time to First Token
Mean TTFT (ms):               26924.13
Median TTFT (ms):             19415.39
P99 TTFT (ms):                84458.36
--- Mean Time per Output Token (excl. 1st token) -----
Mean TPOT (ms):               40.74
Median TPOT (ms):             38.67
P99 TPOT (ms):                76.83
----------------- Inter-Token Latency -----------------
Mean ITL (ms):                40.75
Median ITL (ms):              33.99
P99 ITL (ms):                 87.67
Max ITL (ms):                 788.71
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
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

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
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 288 --random-input-len 3500 --random-output-len 1500 --num-prompts 1088 --random-range-ratio 1
```

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 272
Successful requests: 1088
Benchmark duration (s): 427.04
Total input tokens: 3808000
Total input text tokens: 0
Total input vision tokens: 0
Total generated tokens: 1632000
Total generated tokens (retokenized): 1631408
Request throughput (req/s): 2.55
Input token throughput (tok/s): 8917.16
Output token throughput (tok/s): 3821.64
Peak output token throughput (tok/s): 9304.00
Peak concurrent requests: 313
Total token throughput (tok/s): 12738.79
Concurrency: 239.32
Accept length: 2.83
=== End-to-End Latency ===
Mean E2E Latency (ms): 93931.66
Median E2E Latency (ms): 88251.02
=== Time to First Token ===
Mean TTFT (ms): 16721.52
Median TTFT (ms): 15137.02
P99 TTFT (ms): 42314.45
=== Time per Output Token (excl. 1st token) ===
Mean TPOT (ms): 51.51
Median TPOT (ms): 48.14
P99 TPOT (ms): 109.71
=== Inter-Token Latency ===
Mean ITL (ms): 51.51
Median ITL (ms): 25.60
P95 ITL (ms): 103.61
P99 ITL (ms): 461.44
Max ITL (ms): 35459.74
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

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 576
Successful requests: 576
Benchmark duration (s): 182.26
Total input tokens: 1152000
Total input text tokens: 1152000
Total input vision tokens: 0
Total generated tokens: 1152000
Total generated tokens (retokenized): 1151728
Request throughput (req/s): 3.16
Input token throughput (tok/s): 6320.57
Output token throughput (tok/s): 6320.57
Peak output token throughput (tok/s): 14513.00
Peak concurrent requests: 576
Total token throughput (tok/s): 12641.14
Concurrency: 358.87
Accept length: 2.94

--------End-to-End Latency--------
Mean E2E Latency (ms): 113557.20
Median E2E Latency (ms): 113056.54

--------Time to First Token--------
Mean TTFT (ms): 4049.94
Median TTFT (ms): 4004.61
P99 TTFT (ms): 5926.98

--------Time per Output Token (excl. 1st token)--------
Mean TPOT (ms): 54.78
Median TPOT (ms): 54.67
P99 TPOT (ms): 79.92

--------Inter-Token Latency--------
Mean ITL (ms): 54.78
Median ITL (ms): 38.01
P95 ITL (ms): 82.41
P99 ITL (ms): 152.56
Max ITL (ms): 40909.44
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

export HCCL_BUFFSIZE=1600
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

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

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 480
Successful requests: 480
Benchmark duration (s): 166.51
Total input tokens: 983040
Total input text tokens: 983040
Total input vision tokens: 0
Total generated tokens: 982844
Total generated tokens (retokenized): 982844
Request throughput (req/s): 2.88
Input token throughput (tok/s): 5903.96
Output token throughput (tok/s): 5903.96
Peak output token throughput (tok/s): 13839.00
Peak concurrent requests: 480
Total token throughput (tok/s): 11807.92
Concurrency: 297.23
Accept length: 2.95

---End-to-End Latency---
Mean E2E Latency (ms): 103105.60
Median E2E Latency (ms): 101298.94
Mean TTFT (ms): 4457.97
Median TTFT (ms): 4381.36
P99 TTFT (ms): 6589.31
Mean Time per Output Token (excl. 1st token): 48.19
Median TPOt (ms): 47.40
P99 TPOt (ms): 70.18

---Inter-Token Latency---
Mean ITL (ms): 48.19
Median ITL (ms): 33.67
P99 ITL (ms): 76.04
Max ITL (ms): 33993.29
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

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

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
        echo "${MIX_IP[$i]}"+
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host 127.0.0.1 --port 7439 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 32 --mem-fraction-static 0.8 --max-running-requests 768 \
        --attention-backend ascend --device npu --quantization modelslim --enable-dp-attention \
        --moe-a2a-backend ascend_fuseep --cuda-graph-bs 6 8 10 12 18 24 \
        --dist-init-addr 141.61.105.131:5000 --chunked-prefill-size 32768 --max-prefill-tokens 458880 \
        --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
        --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --prefill-round-robin-balance --enable-dp-lm-head --dtype bfloat16 --tokenizer-worker-num 4
        NODE_RANK=$i
        break
    fi
done

```

#### Benchmark

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 768 --random-input-len 2000 --random-output-len 2000 --num-prompts 768 --random-range-ratio 1
```

```angular2html
Backend: sglang
Traffic request rate: inf
Max request concurrency: 768
Successful requests: 768
Benchmark duration (s): 199.18
Total input tokens: 1572864
Total input text tokens: 0
Total input vision tokens: 0
Total generated tokens (retokenized): 1572864
Request throughput (req/s): 4.06
Input token throughput (tok/s): 8314.05
Output token throughput (tok/s): 8314.05
Peak output token throughput (tok/s): 20655.00
Peak concurrent requests: 768
Total token throughput (tok/s): 16628.10
Concurrency: 529.54
Accept length: 3.20

=== End-to-End Latency ===
Mean E2E Latency (ms): 130442.47
Median E2E Latency (ms): 127654.62

=== Time to First Token ===
Mean TTFT (ms): 27838.48
Median TTFT (ms): 29338.37
P99 TTFT (ms): 48169.79

=== Time per Output Token (excl. 1st token) ===
Mean TPOT (ms): 50.12
Median TPOT (ms): 49.54
P99 TPOT (ms): 72.80

=== Inter-Token Latency ===
Mean ITL (ms): 50.12
Median ITL (ms): 30.66
P99 ITL (ms): 133.63
Max ITL (ms): 36393.04
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

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 1
Successful requests: 10690
Benchmark duration (s): 10.69
Total input tokens: 11000
Total input text tokens: 0
Total generated tokens: 1000
Total generated tokens (retokenized): 1000
Request throughput (req/s): 0.09
Input token throughput (tok/s): 1028.75
Output token throughput (tok/s): 93.52
Peak output token throughput (tok/s): 110.00
Peak concurrent requests: 1
Total token throughput (tok/s): 1122.27
Concurrency: 1.60
Accept length: 4.03
--- End-to-End Latency ---
Mean E2E Latency (ms): 10661.60
Median E2E Latency (ms): 10661.60
--- Time to First Token ---
Mean TTFT (ms): 973.98
Median TTFT (ms): 973.98
P99 TTFT (ms): 973.98
--- Time per Output Token (excl. 1st token) ---
Mean TPOT (ms): 9.70
Median TPOT (ms): 9.70
P99 TPOT (ms): 9.70
--- Inter-Token Latency ---
Mean ITL (ms): 9.70
Median ITL (ms): 9.28
P95 ITL (ms): 13.75
P99 ITL (ms): 14.99
Max ITL (ms): 19.83
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
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 32 \
    --disable-radix-cache \
    --chunked-prefill-size 32768 --max-prefill-tokens 65536 --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 8 16 24 32  --dtype bfloat16

```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32
```

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 32
Successful requests: 32
Benchmark duration (s): 38.48
Total input tokens: 199000
Total input text tokens: 192000
Total input vision tokens: 0
Total generated tokens (retokenized): 48000
Request throughput (req/s): 0.83
Input token throughput (tok/s): 4988.98
Output token throughput (tok/s): 1247.24
Peak output token throughput (tok/s): 2245.00
Peak concurrent requests: 32
Total token throughput (tok/s): 6236.22
Concurrency: 28.44
Accept length: 2.15

=== End-to-End Latency ===
Mean E2E Latency (ms): 34202.32
Median E2E Latency (ms): 33902.93

=== Time to First Token ===
Mean TTFT (ms): 8908.53
Median TTFT (ms): 8557.18
P99 TTFT (ms): 12373.42

=== Time per Output Token (excl. 1st token) ===
Mean TPOT (ms): 16.87
Median TPOT (ms): 16.75
P99 TPOT (ms): 22.71

=== Inter-Token Latency ===
Mean ITL (ms): 16.87
Median ITL (ms): 10.77
P95 ITL (ms): 32.01
P99 ITL (ms): 32.62
Max ITL (ms): 7912.53
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
export HCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 32 \
    --disable-radix-cache \
    --base-gpu-id 4 \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 1 4 6 12 18 24 30 32 --dtype bfloat1

```

#### Benchmark

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --random-range-ratio 1 -max-concurrency 1 --random-output-len 1500 --random-input-len 4096 --num-prompts 4
```

```angular2html
========= Serving Benchmark Result =========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 1
Successful requests: 58.03
Benchmark duration (s): 16000
Total input tokens: 16000
Total input text tokens: 0
Total input vision tokens: 0
Total generated tokens (retokenized): 6000
Request throughput (req/s): 0.07
Input token throughput (tok/s): 275.74
Output token throughput (tok/s): 103.40
Peak output token throughput (tok/s): 132.00
Peak concurrent requests: 2
Total token throughput (tok/s): 379.14
Concurrency: 1.00
Accept length: 2.03

---End-to-End Latency---
Mean E2E Latency (ms): 14502.58
Median E2E Latency (ms): 14752.39

---Time to First Token---
Mean TTFT (ms): 317.68
Median TTFT (ms): 315.90
P99 TTFT (ms): 324.07

---Time per Output Token (excl. 1st token)---
Mean TPOT (ms): 9.46
Median TPOT (ms): 9.63
P99 TPOT (ms): 10.87

---Inter-Token Latency---
Mean ITL (ms): 9.46
Median ITL (ms): 6.64
P95 ITL (ms): 19.78
P99 ITL (ms): 20.73
Max ITL (ms): 33.84
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
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7339 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu   \
    --max-running-requests 1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --tp-size 16 --mem-fraction-static 0.72 --cuda-graph-bs 1 --dtype bfloat16
```

#### Benchmark

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7339 --random-range-ratio 1 --max-concurrency 1 --random-output-len 18000 --random-input-len 4000 --num-prompts 1
```

```angular2html
========= Serving Benchmark Result =========
Max request concurrency: 1
Successful requests: 1
Benchmark duration (s): 50.22
Total input tokens: 18000
Total input text tokens: 18000
Total input vision tokens: 0
Total generated tokens: 4000
Total generated tokens (retokenized): 4000
Request throughput (req/s): 0.02
Input token throughput (tok/s): 358.40
Output token throughput (tok/s): 79.64
Peak output token throughput (tok/s): 100.00
Peak concurrent requests: 1
Total token throughput (tok/s): 438.05
Concurrency: 1.00
Accept length: 2.15
----------------------------------------
---End-to-End Latency---
Mean E2E Latency (ms): 50204.39
Median E2E Latency (ms): 50204.39
---Time to First Token---
Mean TTFT (ms): 1132.62
Median TTFT (ms): 1132.62
P99 TTFT (ms): 1132.62
---Time per Output Token (excl. 1st token)---
Mean TPOT (ms): 12.27
Median TPOT (ms): 12.27
P99 TPOT (ms): 12.27
---Inter-Token Latency---
Mean ITL (ms): 12.27
Median ITL (ms): 7.52
P99 ITL (ms): 27.23
Max ITL (ms): 32.56
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
    --chunked-prefill-size -1 --max-prefill-tokens 65536  \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 4  --mem-fraction-static 0.72 --cuda-graph-bs 16 32 64 68 72 78 --dtype bfloat16
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312
```

```angular2html
Backend:                                   sglang
Traffic request rate:                      inf
Max request concurrency:                   78
Successful requests:                       312
Benchmark duration (s):                    330.60
Total input tokens:                        1092000
Total input text tokens:                   1092000
Total input vision tokens:                 0
Total generated tokens:                    468000
Total generated tokens (retokenized):      467994
Request throughput (req/s):                0.94
Input token throughput (tok/s):            3303.13
Output token throughput (tok/s):           1415.63
Peak output token throughput (tok/s):      2344.00
Peak concurrent requests:                  83
Peak token throughput (tok/s):             4718.76
Concurrency:                               74.73
Accept length:                             2.05
-------------------------------------------
Mean E2E Latency (ms):                     79188.24
Median E2E Latency (ms):                   78524.54
-------------------------------------------
Time to First Token
Mean TTFT (ms):                            5371.45
Median TTFT (ms):                          485.89
P99 TTFT (ms):                             24619.48
-------------------------------------------
Time per Output Token (excl. 1st token)
Mean TPOT (ms):                            49.24
Median TPOT (ms):                          49.50
P99 TPOT (ms):                             66.33
-------------------------------------------
Inter-Token Latency
Mean ITL (ms):                             49.24
Median ITL (ms):                           32.50
P95 ITL (ms):                              127.73
P99 ITL (ms):                              380.22
Max ITL (ms):                              12362.57
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
export DISABLE_EAGLE3_QUANT=1

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 7239 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu  --quantization modelslim  \
    --max-running-requests 120 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size -1 --max-prefill-tokens 49152 \
    --tp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs 54 60 66 72 78 84 90 108 114 120 --dtype bfloat16

```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 120
```

```angular2html
Backend: sglang
Traffic request rate: inf
Max request concurrency: 120
Successful requests: 120
Benchmark duration (s): 121.69
Total input tokens: 240000
Total input text tokens: 240000
Total input vision tokens: 0
Total generated tokens: 240000
Total generated tokens (retokenized): 240000
Request throughput (req/s): 0.99
Input token throughput (tok/s): 1972.30
Output token throughput (tok/s): 1972.30
Peak output token throughput (tok/s): 2983.00
Peak concurrent requests: 120
Total token throughput (tok/s): 3944.60
Concurrency: 111.22
Accept length: 1.69

---End-to-End Latency---
Mean E2E Latency (ms): 112785.77
Median E2E Latency (ms): 112758.61

---Time to First Token---
Mean TTFT (ms): 16144.60
Median TTFT (ms): 17229.95
P99 TTFT (ms): 21366.64

---Time per Output Token (excl. 1st token)---
Mean TPOT (ms): 48.34
Median TPOT (ms): 48.06
P99 TPOT (ms): 55.74

---Inter-Token Latency---
Mean ITL (ms): 48.34
Median ITL (ms): 39.36
P99 ITL (ms): 81.71
Max ITL (ms): 12772.77
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

```angular2html
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 156 --random-input-len 3500 --random-output-len 1500 --num-prompts 624 --random-range-ratio 1
```

```angular2html
Backend: sglang
Traffic request rate: 1nf
Max request concurrency: 156
Successful requests: 624
Benchmark duration (s): 295.64
Total input tokens: 2184000
Total input text tokens: 0
Total input vision tokens: 0
Total generated tokens: 936000
Total generated tokens (retokenized): 935999
Request throughput (req/s): 2.11
Input token throughput (tok/s): 7387.40
Output token throughput (tok/s): 3166.03
Peak output token throughput (tok/s): 5631.00
Peak concurrent requests: 166
Total token throughput (tok/s): 10553.42
Concurrency: 148.81
Accept length: 3.17

---End-to-End Latency---
Mean E2E Latency (ms): 70504.60
Median E2E Latency (ms): 70144.81

---Time to First Token---
Mean TTFT (ms): 4020.19
Median TTFT (ms): 734.31
P99 TTFT (ms): 22554.77

---Time per Output Token (excl. 1st token)---
Mean TPOT (ms): 44.35
Median TPOT (ms): 44.46
P99 TPOT (ms): 62.78

---Inter-Token Latency---
Mean ITL (ms): 44.35
Median ITL (ms): 26.29
P99 ITL (ms): 118.72
P95 ITL (ms): 240.57
Max ITL (ms): 18382.51
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

```angular2html
Backend: sglang
Traffic request rate: 8.0
Max request concurrency: 410
Successful requests: 1640
Benchmark duration (s): 384.95
Total input tokens: 5740000
Total input text tokens: 5740000
Total input vision tokens: 0
Total generated tokens: 2460000
Total generated tokens (retokenized): 2449627
Request throughput (req/s): 4.26
Output token throughput (tok/s): 14911.21
Peak output token throughput (tok/s): 6390.52
Peak concurrent requests: 429
Total token throughput (tok/s): 21301.73
Concurrency: 327.33

---End-to-End Latency---
Mean E2E Latency (ms): 76831.52
Median E2E Latency (ms): 77111.46

---Time to First Token---
Mean TTFT (ms): 4470.95
Median TTFT (ms): 3432.63
P99 TTFT (ms): 17805.87

---Time per Output Token (excl. 1st token)---
Mean TPOT (ms): 48.27
Median TPOT (ms): 49.12
P99 TPOT (ms): 50.17

---Inter-Token Latency---
Mean ITL (ms): 48.27
Median ITL (ms): 45.43
P99 ITL (ms): 128.57
Max ITL (ms): 728.88
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

```angular2html
Backend: sglang
Traffic request rate: 20.0
Max request concurrency: 288
Successful requests: 1152
Benchmark duration (s): 372.58
Total input tokens: 4032000
Total input text tokens: 4932000
Total input vision tokens: 0
Total generated tokens: 1728000
Total generated tokens (retokenized): 1723296
Request throughput (req/s): 3.09
Input token throughput (tok/s): 10821.87
Output token throughput (tok/s): 4637.95
Peak output token throughput (tok/s): 6912.00
Peak concurrent requests: 408
Total token throughput (tok/s): 15459.82
Concurrency: 281.13
End-to-End Latency
Mean E2E Latency (ms): 90923.48
Median E2E Latency (ms): 92344.71
Time to First Token
Mean TTFT (ms): 15466.24
Median TTFT (ms): 14426.93
P99 TTFT (ms): 26498.92
Time per Output Token (excl. 1st token)
Mean TPOT (ms): 50.34
Median TPOT (ms): 49.91
P99 TPOT (ms): 59.31
Inter-Token Latency
Mean ITL (ms): 50.34
Median ITL (ms): 42.35
P95 ITL (ms): 90.51
P99 ITL (ms): 389.41
Max ITL (ms): 24730.36
```

### Qwen3 480B High Throughput 50ms 3

Model: Qwen3 480B

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

DataSets: 3.5K1.5K

TPOT: 50ms

#### Model Deployment

```angular2html
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
--tp 16 --dp-size 4 --mem-fraction-static 0.7 --cuda-graph-bs  16 20
```

#### Benchmark

```angular2html
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 80 --random-input-len 3500 --random-output-len 1500 --num-prompts 320 --random-range-ratio 1 --request-rate 8
```

```angular2html
Backend: sglang
Traffic request rate: 8.0
Max request concurrency: 80
Successful requests: 320
Benchmark duration (s): 319.89
Total input tokens: 1120000
Total input text tokens: 1120000
Total input vision tokens: 0
Total generated tokens (retokenized): 477727
Request throughput (req/s): 1.00
Output token throughput (tok/s): 3501.16
Input token throughput (tok/s): 1500.50
Peak output token throughput (tok/s): 1840.00
Peak concurrent requests: 160
Total token throughput (tok/s): 5001.66
Concurrency: 78.91
----------------- End-to-End Latency -----------------
Mean E2E Latency (ms): 78883.93
Median E2E Latency (ms): 79526.23
----------------- Time to First Token -----------------
Mean TTFT (ms): 6627.14
Median TTFT (ms): 6103.91
P99 TTFT (ms): 11953.96
Mean TPTOT (ms): 48.20
Median TPTOT (ms): 48.10
P99 TPTOT (ms): 52.33
----------------- Inter-Token Latency -----------------
Mean ITL (ms): 48.20
Median ITL (ms): 45.16
P99 ITL (ms): 51.04
P99 ITL (ms): 55.76
Max ITL (ms): 16869.36
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
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 80 --random-output-len 1536 --random-input-len 3584 --num-prompts 160
```

```angular2html
==================== Serving Benchmark Result ====================
Backend: sglang
Traffic request rate: inf
Max request concurrency: 80
Successful requests: 160
Benchmark duration (s): 174.90
Total input tokens: 573440
Total input text tokens: 573440
Total input vision tokens: 0
Total generated tokens: 245760
Total generated tokens (retokenized): 245760
Request throughput (req/s): 0.91
Input token throughput (tok/s): 3278.74
Output token throughput (tok/s): 1405.17
Peak output token throughput (tok/s): 1840.00
Peak concurrent requests: 160
Total token throughput (tok/s): 4683.91
Concurrency: 79.87
------------------- End-to-End Latency -------------------
Mean E2E Latency (ms): 87303.49
Median E2E Latency (ms): 87283.35
------------------- Time to First Token -------------------
Mean TTFT (ms): 10688.91
Median TTFT (ms): 10365.16
P99 TTFT (ms): 19285.56
------------------- Time per Output Token (excl. 1st token) -------------------
Mean TPOT (ms): 49.91
Median TPOT (ms): 49.93
P99 TPOT (ms): 55.63
------------------- Inter-Token Latency -------------------
Mean ITL (ms): 49.91
Median ITL (ms): 44.75
P95 ITL (ms): 45.54
P99 ITL (ms): 46.08
Max ITL (ms): 17803.50
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
    --chunked-prefill-size -1 --max-prefill-tokens 135168  --speculative-draft-model-quantization unquant \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
    --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --tp-size 8 --mem-fraction-static 0.72 --cuda-graph-bs 1 4 8 16 24 28 32  --dtype bfloat16
```

#### Benchmark

We tested it based on the GSM8K dataset.

```shell
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7439 --max-concurrency 32 --random-output-len 1500 --random-input-len 6000 --num-prompts 32
```

```angular2html
-------- Serving Benchmark Result --------
Backend:                          sglang
Traffic request rate:             inf
Max request concurrency:          32
Successful requests:              32
Benchmark duration (s):           40.52
Total input tokens:               192000
Total input text tokens:          192000
Total input vision tokens:        0
Total generated tokens:           48000
Total generated tokens (retokenized): 47984
Request throughput (req/s):       0.79
Input token throughput (tok/s):   4738.46
Output token throughput (tok/s):  1184.62
Peak output token throughput (tok/s): 2045.00
Peak concurrent requests:         32
Total token throughput (tok/s):   5923.08
Concurrency:                      28.60
Accept length:                    2.11

-------- End-to-End Latency --------
Mean E2E Latency (ms):            36214.23
Median E2E Latency (ms):          36358.20

-------- Time to First Token --------
Mean TTFT (ms):                   11544.97
Median TTFT (ms):                 11587.54
P99 TTFT (ms):                    11879.45

-------- Time per Output Token (excl. 1st token) --------
Mean TPOT (ms):                   16.46
Median TPOT (ms):                 16.50
P99 TPOT (ms):                    21.34

-------- Inter-Token Latency --------
Mean ITL (ms):                    16.46
Median ITL (ms):                  11.75
P95 ITL (ms):                     34.80
P99 ITL (ms):                     35.88
Max ITL (ms):                     1991.03
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

#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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

```angular2html
========== Serving Benchmark Result ==========
Backend: sglang
Traffic request rate: inf
Max request concurrency: 1
Successful requests: 4
Benchmark duration (s): 62.33
Total input tokens: 16000
Total input text tokens: 16000
Total input vision tokens: 0
Total generated tokens: 6000
Total generated tokens (retokenized): 6000
Request throughput (req/s): 0.06
Input token throughput (tok/s): 256.71
Output token throughput (tok/s): 96.26
Peak output token throughput (tok/s): 124.00
Peak concurrent requests: 2
Total token throughput (tok/s): 352.97
Concurrency: 1.00
Accept length: 2.04
---------- End-to-End Latency ----------
Mean E2E Latency (ms): 15577.88
Median E2E Latency (ms): 15965.08
---------- Time to First Token ----------
Mean TTFT (ms): 312.47
Median TTFT (ms): 312.07
P99 TTFT (ms): 317.13
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms): 10.18
Median TPOT (ms): 10.44
P99 TPOT (ms): 11.51
---------- Inter-Token Latency ----------
Mean ITL (ms): 10.18
Median ITL (ms): 7.07
P95 ITL (ms): 21.00
Max ITL (ms): 27.61
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
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 78 --random-output-len 1500 --random-input-len 3500 --num-prompts 312
```

```angular2html
========== Serving Benchmark Result ==========
Backend:                              sglang
Traffic request rate:                 inf
Max request concurrency:              78
Successful requests:                  312
Benchmark duration (s):               335.40
Total input tokens:                   1092000
Total input text tokens:              1092000
Total input vision tokens:            0
Total generated tokens:               468000
Total generated tokens (retokenized): 467916
Request throughput (req/s):           0.93
Input token throughput (tok/s):       3255.77
Output token throughput (tok/s):      1395.33
Peak output token throughput (tok/s): 2417.00
Peak concurrent requests:             84
Total token throughput (tok/s):       4651.10
Concurrency:                          73.66
Accept length:                        2.05
---------- End-to-End Latency ----------
Mean E2E Latency (ms):                79186.66
Median E2E Latency (ms):              78608.81
---------- Time to First Token ----------
Mean TTFT (ms):                       5786.98
Median TTFT (ms):                     645.64
P99 TTFT (ms):                        26569.53
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                       48.97
Median TPOT (ms):                     49.06
P99 TPOT (ms):                        67.17
---------- Inter-Token Latency ----------
Mean ITL (ms):                        48.97
Median ITL (ms):                      33.00
P95 ITL (ms):                         131.60
P99 ITL (ms):                         391.27
Max ITL (ms):                         13391.14
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
python3 -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 7239 --max-concurrency 120 --random-output-len 2000 --random-input-len 2000 --num-prompts 120
```

```angular2html
========== Serving Benchmark Result ==========
Backend:                         sglang
Traffic request rate:            inf
Max request concurrency:         120
Successful requests:             120
Benchmark duration (s):          117.18
Total input tokens:              2400000
Total input text tokens:         240000
Total input vision tokens:       0
Total generated tokens:          2400000
Total generated tokens (retokenized): 239960
Request throughput (req/s):      1.02
Input token throughput (tok/s):  2048.10
Output token throughput (tok/s): 2048.10
Peak output token throughput (tok/s): 3133.00
Peak concurrent requests:        120
Total token throughput (tok/s):  4096.20
Concurrency:                     111.43
Accept length:                   1.69
---------- End-to-End Latency ----------
Mean E2E Latency (ms):           108815.31
Median E2E Latency (ms):         108737.30
---------- Time to First Token ----------
Mean TTFT (ms):                  17096.53
Median TTFT (ms):                18159.44
P99 TTFT (ms):                   22681.91
---------- Time per Output Token (excl. 1st token) ----------
Mean TPOT (ms):                  45.88
Median TPOT (ms):                45.71
P99 TPOT (ms):                   53.49
---------- Inter-Token Latency ----------
Mean ITL (ms):                   45.88
Median ITL (ms):                 37.46
P95 ITL (ms):                    77.55
P99 ITL (ms):                    81.31
Max ITL (ms):                    13530.47
```
