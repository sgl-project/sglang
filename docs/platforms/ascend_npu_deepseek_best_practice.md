# Best Practice with DeepSeek Series Models on Ascend NPU

This section describes the best practice data of mainstream LLM models such as DeepSeek on the Ascend NPU. If
you encounter issues or have any questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Low-Latency DeepSeek Series Models

| Model             | Hardware      | Cards | Deploy Mode   | Dataset   | TPOT | Quantization | Configuration                                                                |
|-------------------|---------------|-------|---------------|-----------|------|--------------|------------------------------------------------------------------------------|
| Deepseek-R1       | Atlas 800I A3 | 32    | PD Separation | 6K+1.6K   | 20ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-low-latency-on-a3-separation-mode)      |
| Deepseek-R1       | Atlas 800I A3 | 32    | PD Separation | 3.9K+1K   | 20ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-low-latency-on-a3-separation-mode)      |
| Deepseek-R1       | Atlas 800I A3 | 32    | PD Separation | 3.5K+1.5K | 20ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-low-latency-on-a3-separation-mode)      |
| Deepseek-R1       | Atlas 800I A3 | 32    | PD Separation | 3.5K+1K   | 20ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-low-latency-on-a3-separation-mode)      |
| DeepSeek-V3.2-Exp | Atlas 800I A3 | 32    | PD Separation | 64K+3K    | 30ms | W8A8 INT8    | [Optimal Configuration](#deepseek-v32-exp-low-latency-on-a3-separation-mode) |

### DeepSeek-R1 Low Latency on A3 Separation Mode

#### Deployment with TPOT 20ms on 32 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 6K+1.6K, 3.9K+1K, 3.5K+1.5K, 3.5K+1K

TPOT: 20ms

Quantization: W8A8 INT8

- Launch Prefill and Decode Instance

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
        --port 8001 --trust-remote-code --dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i --tp-size 32 --dp-size 16 \
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

- Launch SGLang Router

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${P_IP[0]}:8000 8998 \
    --prefill http://${P_IP[1]}:8000 8999 \
    --decode http://${D_IP[0]}:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
# 6K+1.6K
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 6000 --random-output-len 1600 --num-prompts 32 --random-range-ratio 1

# 3.9K+1K
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3900 --random-output-len 1000 --num-prompts 768 --random-range-ratio 1 --request-rate 16

# 3.5K+1.5K
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1500 --num-prompts 768 --random-range-ratio 1 --request-rate 16

# 3.5K+1K
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1000 --num-prompts 768 --random-range-ratio 1 --request-rate 16
```

### DeepSeek-V32-Exp Low Latency on A3 Separation Mode

#### Deployment with 64K+3K TPOT 30ms on 32 Cards

- Basic Case Details

Model: DeepSeek-V3.2-Exp-W8A8

Hardware: Atlas 800I A3 32Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 64K+3K

TPOT: 30ms

Quantization: W8A8 INT8

- Launch Prefill Instance

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

- Launch Decode Instance

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

- Launch SGLang Router

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<your_prefill_ip1>:8000 8995 \
    --decode http://<your_decode_ip1>:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 32  --random-input-len 64000 --random-output-len 3000 --num-prompts 64 --random-range-ratio 1
```

## High-Throughput DeepSeek Series Models

| Model       | Hardware      | Cards | Deploy Mode   | Dataset   | TPOT | Quantization | Configuration                                                               |
|-------------|---------------|-------|---------------|-----------|------|--------------|-----------------------------------------------------------------------------|
| Deepseek-R1 | Atlas 800I A3 | 16    | PD Separation | 3.5K+1.5K | 50ms | W4A8 INT8    | [Optimal Configuration](#deepseek-r1-high-throughput-on-a3-separation-mode) |
| Deepseek-R1 | Atlas 800I A3 | 32    | PD Separation | 3.5K+1.5K | 50ms | W8A8 INT8    | [Optimal Configuration](#deepseek-r1-high-throughput-on-a3-separation-mode) |
| Deepseek-R1 | Atlas 800I A3 | 16    | PD Separation | 2K+2K     | 50ms | W4A8 INT8    | [Optimal Configuration](#deepseek-r1-high-throughput-on-a3-separation-mode) |
| Deepseek-R1 | Atlas 800I A3 | 8     | PD Mixed      | 3.5K+1.5K | 50ms | W4A8 INT8    | [Optimal Configuration](#deepseek-r1-high-throughput-on-a3-mixed-mode)      |
| Deepseek-R1 | Atlas 800I A3 | 8     | PD Mixed      | 2K+2K     | 50ms | W4A8 INT8    | [Optimal Configuration](#deepseek-r1-high-throughput-on-a3-mixed-mode)      |

### DeepSeek-R1 High Throughput on A3 Separation Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 16 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 16Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W4A8 INT8

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

- Launch SGLang Router

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${P_IP[0]}:8000 8998 \
    --decode http://${D_IP[0]}:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 384  --random-input-len 3500 --random-output-len 1500 --num-prompts 1536 --random-range-ratio 1
```

#### Deployment with 3.5K+1.5K TPOT 50ms on 32 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 32Card

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

- Launch SGLang Router

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${P_IP[0]}:8000 8998 \
    --prefill http://${P_IP[1]}:8000 8999 \
    --decode http://${D_IP[0]}:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 768  --random-input-len 3500 --random-output-len 1500 --num-prompts 3072 --random-range-ratio 1 --request-rate 16
```

#### Deployment with 2K+2K TPOT 50ms on 16 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 16Card

DeployMode: PD Separation

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W4A8 INT8

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

- Launch SGLang Router

```shell
export SGLANG_DP_ROUND_ROBIN=1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${P_IP}:8000 8998 \
    --decode http://${D_IP[0]}:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
```

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6688 --max-concurrency 400  --random-input-len 2048 --random-output-len 2048 --num-prompts 3200 --random-range-ratio 1 --request-rate 8
```

### DeepSeek-R1 High Throughput on A3 Mixed Mode

#### Deployment with 3.5K+1.5K TPOT 50ms on 8 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

Quantization: W4A8 INT8

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 144  --random-input-len 3500 --random-output-len 1500 --num-prompts 576 --random-range-ratio 1
```

#### Deployment with 2K+2K TPOT 50ms on 8 Cards

- Basic Case Details

Model: Deepseek R1

Hardware: Atlas 800I A3 8Card

DeployMode: PD Mixed

Dataset: random

Input Output Length: 2K+2K

TPOT: 50ms

Quantization: W4A8 INT8

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

- Benchmark

We tested it based on the `RANDOM` dataset.

```shell
python -m sglang.bench_serving --dataset-name random --backend sglang --host 127.0.0.1 --port 6699 --max-concurrency 256  --random-input-len 2048 --random-output-len 2048 --num-prompts 1024 --random-range-ratio 1
```
