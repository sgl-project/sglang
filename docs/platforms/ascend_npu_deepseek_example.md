## DeepSeek examples

### Running DeepSeek-V3

#### Running DeepSeek in PD mixed mode on 1 x Atlas 800I A3.

W4A8 Model weights could be found [here](https://modelers.cn/models/Modelers_Park/DeepSeek-R1-0528-w4a8).

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=1600

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

#npu acceleration operator
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 16 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --watchdog-timeout 9000 \
    --cuda-graph-bs 8 16 24 28 32 \
    --mem-fraction-static 0.68 \
    --max-running-requests 128 \
    --context-length 8188 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 16384 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-dp-attention \
    --dp-size 4 \
    --enable-dp-lm-head \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dtype bfloat16
```

#### Running DeepSeek with PD disaggregation mode on 2 x Atlas 800I A3.

W4A8 Model weights could be found [here](https://modelers.cn/models/Modelers_Park/DeepSeek-R1-0528-w4a8).

1. Prefill:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"

#Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=1536

#npu acceleration operator
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export TASK_QUEUE_ENABLE=2

python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --host $PREFILL_HOST_IP \
    --port 8000 \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port 8996 \
    --disaggregation-transfer-backend ascend \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 16 \
    --mem-fraction-static 0.6 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --load-balance-method round_robin \
    --max-running-requests 8 \
    --context-length 8192 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-prefill-tokens 28680 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dp-size 2 \
    --enable-dp-attention \
    --disable-shared-experts-fusion \
    --dtype bfloat16
```

2. Decode:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:<PORT>"

#Deepep communication settings
export HCCL_BUFFSIZE=720
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=88

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

#npu acceleration operator
unset TASK_QUEUE_ENABLE
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

# suggest max-running-requests <= max-cuda-graph-bs * dp_size, Because when this value is exceeded, performance will significantly degrade.
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --host $DECODE_HOST_IP \
    --port 8001 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 16 \
    --dp-size 16 \
    --mem-fraction-static 0.8 \
    --max-running-requests 352 \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --prefill-round-robin-balance \
    --moe-a2a-backend deepep \
    --enable-dp-attention \
    --deepep-mode low_latency \
    --enable-dp-lm-head \
    --cuda-graph-bs 8 10 12 14 16 18 20 22 \
    --disaggregation-transfer-backend ascend \
    --watchdog-timeout 9000 \
    --context-length 8192 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --tokenizer-worker-num 4
```

3. SGLang Model Gateway (former Router)

```shell
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<PREFILL_HOST_IP>:8000 8996 \
    --decode http://<DECODE_HOST_IP>:8001 \
    --host 127.0.0.1 \
    --port 6688
```

#### Running DeepSeek with PD disaggregation on 4 x Atlas 800I A3.

W8A8 Model weights could be found [here](https://modelers.cn/models/State_Cloud/Deepseek-R1-bf16-hfd-w8a8).

1. Prefill & Decode:

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

2. SGLang Model Gateway (former Router):

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

#### test gsm8k

```python
from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval

def gsm8k():
    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=32,
        host=f"http://127.0.0.1",
        port=6688,
    )
    metrics = run_eval(args)
    print(f"{metrics=}")
    print(f"{metrics['accuracy']=}")
if __name__ == "__main__":
    gsm8k()
```
