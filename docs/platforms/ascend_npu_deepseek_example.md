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
export ENABLE_MOE_NZ=1

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
export ENABLE_MOE_NZ=1
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

1. Prefill:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<P_HOST_IP[0]>:<PORT>"

#Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=1536

#npu acceleration operator
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1
export TASK_QUEUE_ENABLE=2

#Please list all host ips of Prefill instance
P_HOST_IP=('xx.xx.xx.xx' 'xx.xx.xx.xx')

for i in "${!P_HOST_IP[@]}";
do
  python -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --host ${P_HOST_IP[$i]} \
      --port 8000 \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port $((8996+$i)) \
      --disaggregation-transfer-backend ascend \
      --trust-remote-code \
      --nnodes 1 \
      --node-rank 0 \
      --tp-size 16 \
      --mem-fraction-static 0.81 \
      --attention-backend ascend \
      --device npu \
      --max-running-requests 8 \
      --context-length 8192 \
      --disable-radix-cache \
      --chunked-prefill-size -1 \
      --max-prefill-tokens 28680 \
      --moe-a2a-backend deepep \
      --deepep-mode normal \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 1 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 2 \
      --dp-size 2 \
      --enable-dp-attention \
      --disable-shared-experts-fusion \
      --dtype bfloat16
done
```

2. Decode:

```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#memfabric config store
export ASCEND_MF_STORE_URL="tcp://<P_HOST_IP[0]>:<PORT>"

#Deepep communication settings
export HCCL_BUFFSIZE=600
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=78

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

#npu acceleration operator
unset TASK_QUEUE_ENABLE
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export ENABLE_MOE_NZ=1

#please list all host ips of Prefill instance
D_HOST_IP=('xx.xx.xx.xx' 'xx.xx.xx.xx')

for i in "${!D_HOST_IP[@]}";
do
  python -m sglang.launch_server
      --model-path ${MODEL_PATH} \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend ascend \
      --host ${D_HOST_IP[$i]} \
      --port 8001 \
      --trust-remote-code \
      --dist-init-addr ${D_HOST_IP[0]}:5000 \
      --nnodes 2 \
      --node-rank $i \
      --tp-size 32 \
      --dp-size 32 \
      --mem-fraction-static 0.8 \
      --max-running-requests 832 \
      --attention-backend ascend \
      --device npu \
      --moe-a2a-backend deepep \
      --enable-dp-attention \
      --deepep-mode low_latency \
      --enable-dp-lm-head \
      --cuda-graph-bs  8 10 12 14 16 18 20 22 24 26 \
      --watchdog-timeout 9000 \
      --context-length 8192 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 2 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 3  \
      --tokenizer-worker-num 4 \
      --disable-shared-experts-fusion \
      --dtype bfloat16
done
```

3. SGLang Model Gateway (former Router):

```shell
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://<P_HOST_IP[0]>:8000 8996 \
    --prefill http://<P_HOST_IP[1]>:8000 8997 \
    --decode http://<D_HOST_IP[0]>:8001 \
    --host 127.0.0.1 \
    --port 6688
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
