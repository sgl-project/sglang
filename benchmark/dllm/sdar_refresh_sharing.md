# SDAR block-refresh sharing

Set `SGLANG_SDAR_REFRESH_PAIR=1` to share a completed-block KV refresh with the next block's first denoising pass.
The default is disabled.

Each layer writes the completed block's refreshed KV before next-block attention reads it.
The gate/up projection processes both blocks together.
QKV, attention-output, and FFN-down projections keep their original row counts and use two ordered CUDA streams.
The decoding threshold, token selection rule, and terminal refresh stay unchanged.

## Supported configuration

- Dense SDAR, BF16 weights and KV, TP=PP=1.
- LowConfidence with FDFO and CUDA Graphs, including an N1 decode capture.
- FlashInfer ragged attention, full attention, page size one, and a static KV pool.
- One active request at the eligible refresh boundary.
- Resident weights, without memory saver, CPU offloading, or model debug hooks.
- Native diffusion decoding, without speculative decoding.

The server validates these settings and captures the paired graph before readiness.
Online weight updates are rejected while sharing is enabled.
Replacing the native decode graph disables sharing and discards pending predictions before their next forward.

Larger batches use native execution. Pending N1 predictions are discarded before an incompatible consuming batch.
Requests for logprobs, hidden states, LoRA, input embeddings, or explicit stop rules also use native execution.
EOS, output caps, and context limits prevent work beyond eligible block boundaries.
Retraction clears pending predictions.

The adapter borrows one scratch KV block and frees it after each paired call.
Scratch slots never enter the scheduler's request table.
There is one model instance. The additional CUDA Graph and attention metadata consume extra memory.

## Run

Use the block size specified by the checkpoint.
For B16:

```bash
cat > /tmp/sdar-b16.yaml <<'YAML'
block_size: 16
threshold: 0.95
YAML

SGLANG_SDAR_REFRESH_PAIR=1 SGLANG_FLASHINFER_USE_PAGED=0 \
python -m sglang.launch_server \
  --model-path JetLM/SDAR-8B-Chat-b16 \
  --dtype bfloat16 --tp 1 --attention-backend flashinfer \
  --dllm-algorithm LowConfidence --dllm-fdfo \
  --dllm-algorithm-config /tmp/sdar-b16.yaml \
  --disable-overlap-schedule --disable-radix-cache \
  --cuda-graph-bs-decode 1 2 4 --max-running-requests 4
```

For the native reference, use the same command with `SGLANG_SDAR_REFRESH_PAIR=0`.
Use identical model revisions, prompts, output caps, thresholds, and concurrency.

The first eligible transition compares both blocks against native CUDA Graph execution.
The check requires finite logits, `atol=0.01`, `rtol=0.01`, equal greedy IDs, and equal next-block confidence decisions.
A failed comparison disables sharing and uses native execution. Unexpected execution errors still propagate.
This first-transition check does not guarantee exact outputs for every request.

Run the first-transition comparison before measuring steady-state performance.
Check complete output IDs and finish reasons before comparing timings.
Measure graph memory separately from the temporary KV block.

## CPU regression tests

```bash
CUDA_VISIBLE_DEVICES=9999 PYTHONPATH=python \
python test/registered/unit/dllm/test_sdar_refresh_pair.py -v
```

These tests cover request lifecycle, token metadata, attention ordering, allocation failure, graph qualification, and native fallback.
They do not replace GPU checkpoint tests or serving measurements.

## GPU regression test

```bash
PYTHONPATH=python python test/registered/dllm/test_sdar_refresh_sharing.py -v
```

This registered NVIDIA test compares full outputs from the original B4 checkpoint at threshold 0.95.
It checks native EOS behavior, repeated requests, and successful numerical qualification.
