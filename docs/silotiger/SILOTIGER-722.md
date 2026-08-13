# SILOTIGER-722: dense MiniMax-M3 MXFP8 AITER/FlyDSL path

## Scope

This commit dispatches exact MiniMax-M3 dense MXFP8 signatures to the paired
AITER MX-scale preshuffle GEMM while preserving canonical checkpoint tensors
for fallback.

The selected path consists of:

```text
BF16 activation
  -> canonical MXFP8 quantization with direct FlyDSL scale-layout stores
  -> tuned FlyDSL MXFP8 GEMM
  -> BF16 output
```

Unknown signatures, unsupported topologies, and eager decode remain on the
canonical Triton path.

## Paired AITER and AOT

The image must contain the AITER SILOTIGER-722 commit and its 147-row model
configuration. Before serving, run:

```bash
cd /sgl-workspace/aiter
PYTHONPATH=. python3 -m aiter.aot.flydsl.mxscale_preshuffle \
  --csv aiter/configs/model_configs/minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv
```

Require `147 ok, 0 failed`. Without the paired AITER symbols
`get_mxscale_preshuffle_config` and `gemm_mxscale_preshuffle`, the dense AITER
backend is not usable.

## Runtime arguments

```bash
export SGLANG_USE_AITER=1

python -m sglang.launch_server \
  --model-path <MiniMax-M3-MXFP8-snapshot> \
  --quantization mxfp8 \
  --dtype bfloat16 \
  --tp 4 \
  --ep-size 4 \
  --fp8-gemm-backend aiter
```

Decode FlyDSL selection requires enabled CUDA graphs and an exact captured
bucket. The covered decode buckets are:

```text
1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64
```

The exact eager-prefill values include the original power-of-two signatures and
`M=8320`. Other prefill tails stay on Triton until separately measured.

Restart the server when changing dense backends. Load-time preshuffle creates a
second weight/scale layout; it cannot be safely toggled in process.

## Memory and capacity

Keeping both canonical and preshuffled weights costs roughly 2.6 GiB per TP4
rank for this model. Pointer-stable CUDA-graph buffers add further memory.
Measure graph-capture memory and `max_total_num_tokens` in the deployment image;
the previous stack observed approximately 3% lower KV capacity.

Do not trade away canonical tensors merely to recover memory: they are required
for unknown-M, eager, and topology fallbacks.

## Validation

```bash
pytest -q python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py
pytest -q test/registered/unit/layers/quantization/test_minimax_m3_mxfp8_aiter.py
```

The serving gate must cover C=`1,2,4,8,16,32,64`, retain client
TTFT/TPOT/ITL/p99 JSON, and compare absolute throughput against the legacy
champion with the same model snapshot and chat template.
