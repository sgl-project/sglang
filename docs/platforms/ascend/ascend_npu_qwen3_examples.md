## Qwen3 examples

### Running Qwen3

#### Running Qwen3-32B on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-32B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8
```

#### Running Qwen3-32B on 1 x Atlas 800I A3 with Qwen3-32B-Eagle3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-32B)

Speculative model weights could be found [here](https://huggingface.co/Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8 \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path Qwen/Qwen3-32B-Eagle3 \
   --speculative-num-steps 1 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 2
```

#### Running Qwen3-30B-A3B MOE on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-30B-A3B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-30B-A3B \
   --mem-fraction-static 0.8
```

#### Running Qwen3-235B-A22B-Instruct-2507 MOE on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
   --tp-size 16 \
   --trust-remote-code \
   --attention-backend ascend \
   --device npu \
   --watchdog-timeout 9000 \
   --mem-fraction-static 0.8
```

#### Running Qwen3-235B-A22B-Instruct-2507 with 256K long sequence on 2 x Atlas 800I A3 without CP

This example uses **PD disaggregation** for long-sequence inference and keeps **context parallel disabled**.

Set the shared environment variables on both nodes first:

```shell
export ASCEND_USE_FIA=1
export SGLANG_SET_CPU_AFFINITY=1
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:12345"
export HCCL_SOCKET_IFNAME=<NETWORK_IFACE>
export GLOO_SOCKET_IFNAME=<NETWORK_IFACE>

MODEL_PATH=/root/.cache/modelscope/hub/models/zcgy26/Qwen3-235B-A22B-Instruct-2507-w8a8
```

**Prefill node:**

```shell
export ASCEND_LAUNCH_BLOCKING=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=1500
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=128
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

python3 -m sglang.launch_server \
   --model-path ${MODEL_PATH} \
   --disaggregation-mode prefill \
   --disaggregation-transfer-backend ascend \
   --disaggregation-bootstrap-port 8995 \
   --attention-backend ascend \
   --disable-radix-cache \
   --quantization modelslim \
   --chunked-prefill-size -1 \
   --skip-server-warmup \
   --device npu \
   --tp-size 16 \
   --mem-fraction-static 0.45 \
   --max-running-requests 1 \
   --host <PREFILL_HOST_IP> \
   --port 8000 \
   --dist-init-addr <PREFILL_HOST_IP>:5000 \
   --nnodes 1 \
   --node-rank 0 \
   --moe-a2a-backend deepep \
   --deepep-mode normal
```

**Decode node:**

```shell
export SGLANG_DEEPEP_BF16_DISPATCH=0
export HCCL_BUFFSIZE=4000
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=4096
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16

python3 -m sglang.launch_server \
   --model-path ${MODEL_PATH} \
   --disaggregation-mode decode \
   --disaggregation-transfer-backend ascend \
   --attention-backend ascend \
   --mem-fraction-static 0.8 \
   --disable-cuda-graph \
   --device npu \
   --disable-radix-cache \
   --quantization modelslim \
   --chunked-prefill-size 8192 \
   --skip-server-warmup \
   --tp-size 16 \
   --max-running-requests 1 \
   --host <DECODE_HOST_IP> \
   --port 8232 \
   --moe-a2a-backend deepep \
   --deepep-mode low_latency \
   --disable-overlap-schedule
```

**Router:**

```shell
python3 -m sglang_router.launch_router \
   --pd-disaggregation \
   --policy cache_aware \
   --prefill http://<PREFILL_HOST_IP>:8000 8995 \
   --decode http://<DECODE_HOST_IP>:8232 \
   --host <ROUTER_HOST_IP> \
   --port 6689 \
   --prometheus-port 29010
```

#### Running Qwen3-VL-8B-Instruct on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --enable-multimodal \
   --attention-backend ascend \
   --mm-attention-backend ascend_attn \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-VL-8B-Instruct \
   --mem-fraction-static 0.8
```
