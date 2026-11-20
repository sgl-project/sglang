# Experimental Ascend TP2 with KT CPU experts

This implementation adds Ascend NPU support to KT CPU-expert offload and can
opt into TP=2/EP=1 with
`KT_NPU_EXPERIMENTAL_TP2=1`. Without the switch, NPU KT offload keeps its
TP=1/EP=1 requirement. Other TP/EP sizes remain unsupported. The existing
rank-0 CPU executor and NPU expert partition/reduction paths are reused.

## Tested environment

- Dual-socket Kunpeng 920 and two Ascend 910B4-1 NPUs (64 GiB each).
- Linux aarch64, CANN 9.0.0, torch_npu 2.10.0.
- DeepSeek-V4-Flash W8A8 checkpoint and per-layer MXFP4 CPU expert GGUFs.
- KT `c9873312643b615a717832946fba0346e69ab035` and SGLang base
  `ff46d660d16cbf11ef393c80cc6b3ab7f6259780` plus the original opt-in patch.

The implementation adapts the Apache-2.0 Ascend work by
[Pan-Boyi and contributors](https://github.com/Pan-Boyi/sglang/tree/ff46d660d16cbf11ef393c80cc6b3ab7f6259780).
It retains upstream's standard dispatcher and NPU runner instead of importing
the fork's older dispatcher hooks, streaming prefill, or expert-placement options.

**The hardware results below are from that fork plus the original TP2 patch.
This port to official SGLang main has CPU-only tests and has not been run on NPU
hardware. The command below is a reproduction starting point for the port,
not a hardware-verified upstream recipe.**

Set up an Ascend SGLang environment and matching Ascend-enabled kt-kernel first.
The CPU wrapper must expose `gpu_experts_mask`, `swiglu_limit`,
`copy_inputs_to_cpu_buffers`, `run_pinned_forward_sync`, and
`copy_forward_output_to_device`; see
[KT's Ascend tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md).
Set `MODEL_PATH` to your local W8A8 checkpoint directory and
`CPU_EXPERT_TEMPLATE` to your per-layer GGUF path template.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export KT_NPU_EXPERIMENTAL_TP2=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
export SGLANG_SET_CPU_AFFINITY=0
export KT_PREFILL_STREAM=0 KT_DYNAMIC_RESIDENT=0 KT_SIDE_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True TASK_QUEUE_ENABLE=1
ulimit -n 65536

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --device npu --attention-backend ascend \
  --tensor-parallel-size 2 --expert-parallel-size 1 --moe-a2a-backend none \
  --page-size 128 --quantization compressed-tensors \
  --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code \
  --disable-radix-cache --mem-fraction-static 0.81 \
  --context-length 8192 --max-prefill-tokens 8191 --chunked-prefill-size 512 \
  --max-total-tokens 16384 --watchdog-timeout 600 \
  --kt-method LLAMAFILE --kt-num-gpu-experts 32 \
  --kt-weight-path "$CPU_EXPERT_TEMPLATE" \
  --kt-threadpool-count 4 --kt-cpuinfer 96 \
  --max-running-requests 1 --host 127.0.0.1 --port 18082
```

Add `--disable-cuda-graph` for eager execution. The tested graph mode used
batch=1 decode graphs; prefill graphs were not enabled. An explicit
`--max-total-tokens 16384` avoided an automatic KV/indexer sizing OOM in the
tested environment. This is a capacity workaround, not a memory-budget fix.
The 8192 context setting is not evidence of an 8192-token validation run.

## Validation and limits

Two-rank HCCL, eager and decode-graph smoke requests, SSE, and 24 fixed-length
generation requests completed. The 43 MoE layers reported intermediate
dimensions 1024 per rank / 2048 full, with CPU ownership only on rank 0.

Numerical correctness remains unresolved: a fixed-history TP1/TP2 probe found
5 different top-1 predictions among 72 input positions, a maximum absolute
logit difference of 3.75, and maximum total variation distance of 0.33250.
Within each TP setting, repeat runs and eager/graph logits matched exactly.
Independent task-quality evaluation has not been completed. Do not assume
the observed cross-TP differences are expected rounding error.

Validation does not cover concurrent requests, long context, other models or
TP/EP sizes, or graceful shutdown. The upstream port has no hardware validation
yet. Resident experts must be a contiguous prefix; expert placement updates,
A2A dispatch, and streaming prefill are outside this implementation's scope.

See the [hardware report and original reproduction patch](https://github.com/kvcache-ai/ktransformers/pull/2157#issuecomment-6810709188).
