# Running Qwen3-32B

Running Qwen3-32B on 4xAtlas 800I A3.
Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-32B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --device npu --attention-backend ascend --trust-remote-code --tp-size 4 --model-path Qwen/Qwen3-32B --port 30111 --mem-fraction-static 0.8
```

# Running Qwen3-MoE

Running Qwen3-30B-A3B on 4xAtlas 800I A3.
Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-30B-A3B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export ENABLE_ASCEND_MOE_NZ=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --device npu --attention-backend ascend --trust-remote-code --tp-size 4 --model-path Qwen/Qwen3-30B-A3B --port 30111 --mem-fraction-static 0.8
```

Running Qwen3-235B-A22B on 16xAtlas 800I A3.
Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export ENABLE_ASCEND_MOE_NZ=1

python -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 --tp-size 16 --trust-remote-code --attention-backend ascend --device npu --watchdog-timeout 9000 --port 30111 --mem-fraction-static 0.8
```
