# CSGMV compact expand and split-K shrink

Compact expand launches only the output tiles needed by each projection slice.
The LoRA backend passes the existing CPU slice offsets to the kernel, so dispatch
requires no device-to-host copy. Callers without CPU offsets use the rectangular
grid with an early return for empty tiles.

Split-K shrink divides the input-feature reduction between CTAs and combines their
FP32 partials in a second kernel. This helps small batches whose rank dimension
provides too few output tiles to fill the GPU.

Enable it with `SGLANG_CSGMV_SPLIT_K=1` and `--lora-backend csgmv`. The initial policy
covers BF16 rank-32 adapters, at most 128 token rows, and the default 16×16×256
shrink tile: eight splits for input width 4,096 and sixteen for width 12,288.
Other shapes use the original shrink kernel. Deterministic inference disables
split-K because changing the reduction order changes floating-point rounding.

## Tests and kernel timings

```bash
PYTHONPATH=python pytest -q \
  test/registered/kernels/ops/gemm/test_csgmv_split_k.py \
  test/registered/kernels/ops/gemm/test_chunked_sgmv_cuda_graph.py

PYTHONPATH=python python benchmark/kernels/lora_csgmv/bench_csgmv_split_k.py \
  --tokens 32 128 8192 --adapters 8
```

The benchmark compares shrink plus expand under CUDA graphs. It excludes routing
and host dispatch, includes the split-K reduction, and reuses an output buffer
that accumulates during timing. It prints JSON lines with GPU and workload details.

## Upstream kernel measurement

The port passed 34 GPU tests on H200. With eight adapters, BF16 rank 32 and
32 token rows, shrink-plus-expand timings were:

| Projection | Early-return baseline | Compact + split-K | Speedup |
|---|---:|---:|---:|
| qkv | 20.33 µs | 15.25 µs | 1.33× |
| gate_up | 27.71 µs | 22.27 µs | 1.24× |
| down | 32.17 µs | 10.98 µs | 2.93× |

At 128 rows, down projection improved 2.29×. At 8,192 rows, split-K is disabled;
compact QKV expand improved the combined pair by about 7%. These are CUDA-graph
kernel timings and exclude CPU dispatch.

## Earlier full-model experiment

A prototype of these kernels was measured on SGLang revision
`d050d06437d96196fc68d5b4e5c246408790d537`, before this upstream port. It used
Qwen3.5-9B on one H200, BF16, TP=1, CUDA graphs, eight synthetic nonzero rank-32
adapters, 32 concurrent requests, and 512-token prompts. Adapter targets were
q/k/v/o and gate/up/down. Output lengths were fixed with `ignore_eos=True`.

The baseline included the empty-tile early-return patch. Requests went through
localhost SGLang HTTP with output logprobs enabled. Runs alternated baseline,
prototype, prototype, baseline, with server restarts, warmup and cache flushes.

| Output tokens/request | Baseline repeats | Prototype repeats |
|---:|---:|---:|
| 4,096 | 40.11 / 43.21 s | 33.99 / 34.37 s |
| 8,192 | 83.38 / 83.75 s | 71.32 / 71.27 s |

The 8K result gives 17.2% higher throughput. One 4K baseline had intermittent
slowdowns; using its faster repeat gives a conservative 17.4% improvement.
All 256 requests completed with the requested lengths and zero retractions.

The prototype's compact expand matched the original bit-for-bit in 24 kernel
cases. All 96 split-K variants passed BF16 checks against FP32 matmul; the largest
observed shrink difference from the original was 0.03125. These measurements use
synthetic adapters and do not establish task accuracy. The upstream port adds
CPU-offset plumbing and a reduction that masks inactive rows and rank columns;
its full-model throughput still needs to be remeasured.
