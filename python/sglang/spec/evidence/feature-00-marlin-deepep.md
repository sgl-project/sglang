Status: implemented

# Marlin + DeepEP evidence

TL;DR: Numerical and graph tests pass across five quantization families.
Actual AWQ server throughput improves with DeepEP at eight-GPU EP scale;
the two-GPU results remain workload-dependent. All repeated results are below.

## Environment

Measured 2026-09-05 on OSS SGLang main
`756d0e0a851c0aee706670affa90c3c47e317d15` plus this change.
NVIDIA H200 GPUs; driver 595.71.05; CUDA toolkit 13.0; Python 3.12;
PyTorch 2.13.0+cu130; Triton 3.7.1; sgl-deep-ep 0.1.2;
sglang-kernel 0.4.6.post1; apache-tvm-ffi 0.1.11; pytest 9.1.1.
Native Marlin specializations compile on first use.

## Reproduction

Run from the repository root with the dependencies above. These commands use
this machine's Python environment and assign independent GPUs to each suite.
The checked-in test files are the reproduction scripts.

```bash
PYTHONPATH=python /opt/sglang/bin/python -m pytest test/registered/unit/layers/moe/test_marlin_deepep.py test/registered/unit/layers/moe/test_w4afp8_deepep_dtype.py -q
PYTHONPATH=python /opt/sglang/bin/python -m pytest test/registered/kernels/ops/moe/test_marlin_deepep.py -xq
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python /opt/sglang/bin/python -m pytest test/registered/kernels/ops/moe/test_moe_wna16_marlin.py -k 'nvfp4_non_gated_matches_dequant_reference or large_non_ep_schedule' -xq
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=python SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32 /opt/sglang/bin/torchrun --standalone --nproc-per-node=2 test/registered/ep/test_marlin_deepep.py
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=python SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32 /opt/sglang/bin/python test/registered/ep/test_marlin_deepep_server.py
```

## Correctness

- Host configuration, dispatcher dtype, and quantization wrappers: 22 passed.
- Dispatch adapters: 31 passed, covering active one-token inputs, invalid
  routes, biased experts, supported activations, and graph replay with changing
  valid counts (including all-zero counts).
- Standard Marlin regressions: 6 passed (26 deselected) in 251.55 seconds,
  including NVFP4 with/without bias and large non-EP scheduling.
- Distributed GPTQ4, GPTQ8, AWQ, MXFP4, NVFP4: 40 format/batch/mode
  combinations passed. Twenty low-latency captures each replay twice,
  including uneven batches, rank-local expert skew, one empty rank, and
  both empty ranks. The same AUTO dispatcher switches modes.
- Original smoke server normal, low_latency, auto: 3 passed in 148.09 seconds. Each uses
  two tensor/expert-parallel ranks, prefill, and four decode tokens for
  single and unequal two-request batches with decode graphs enabled.
  Pytest reported 18 dependency/process-cleanup warnings, including a
  multiprocessing resource tracker restart; all engine assertions passed.

`python/sglang/test/marlin_deepep_utils.py` generates deterministic quantized
weights (seed 42) and independent dequantized references. Integer fixtures
exercise group size 128, GPTQ activation ordering, and AWQ zero points.
FP4 fixtures use native repacking and checkpoint-normalized group/global
scales. Reference comparisons use BF16 rounding at GEMM/activation boundaries:
GPTQ/AWQ/MXFP4 rtol=0.04, atol=0.04; NVFP4 rtol=0.05, atol=0.25,
plus relative L2 error below 2% for every nonempty comparison. Direct parity
between two independently rounded kernel outputs allows twice the per-format
absolute/relative error budget. Standard masked routes produce zero outputs.
Low-latency communication padding is unspecified: the adapter tests poison
input padding, and distributed tests poison output padding with NaNs before
both eager and captured combine, verifying that only valid rows contribute.

## Initial component latency (before throughput optimization)

Recorded at `46d7a52a68`, before the valid-row optimizations.

Rank-zero GPU event measurements, milliseconds, average of five iterations
after one warmup. Synthetic inputs have 17/20 tokens per rank, hidden size
4096, intermediate size 128, four experts, top-k two, BF16 activations, and
DeepEP capacity 32. The normal and low-latency paths represent prefill and
decode communication respectively; timings use eager execution and include
launch gaps. The expert column includes adapter work. Total spans dispatch
through combine, not an entire server request.

| Format | Mode | Dispatch ms | Experts ms | Combine ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| gptq4 | normal | 0.208 | 0.389 | 0.163 | 0.759 |
| gptq4 | low_latency | 0.070 | 0.285 | 0.065 | 0.419 |
| gptq8 | normal | 0.136 | 0.252 | 0.106 | 0.494 |
| gptq8 | low_latency | 0.064 | 0.284 | 0.060 | 0.409 |
| awq | normal | 0.119 | 0.246 | 0.091 | 0.457 |
| awq | low_latency | 0.069 | 0.276 | 0.056 | 0.401 |
| mxfp4 | normal | 0.119 | 0.267 | 0.078 | 0.464 |
| mxfp4 | low_latency | 0.062 | 0.326 | 0.026 | 0.414 |
| nvfp4 | normal | 0.115 | 0.264 | 0.084 | 0.462 |
| nvfp4 | low_latency | 0.062 | 0.283 | 0.054 | 0.400 |

## Actual HTTP serving throughput

Checkpoint: [QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ](https://huggingface.co/QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ/tree/c58857a7f41c0920f73d1b56678640f9c02017d7),
revision `c58857a7f41c0920f73d1b56678640f9c02017d7`. This loads real pretrained
weights and tokenizer, not the dummy smoke model. It has 48 layers, hidden
size 2048, MoE intermediate size 768, 128 experts, top-k eight, and AWQ4
with group size 128 and zero points.

Each pair runs sequentially on the same H200 GPUs in one host (NV18 links
between every GPU pair). TP=EP=DP=GPU count, DP attention enabled, Marlin
BF16, Triton attention, decode graphs enabled, prefill graphs disabled,
normal scheduling overlap enabled, and DeepEP auto mode. The script records
complete server commands and benchmark JSON, including resolved server args.
Measured GPU pairs are reserved for their server. One short validation run
on GPU 2 overlapped the end of the last two-GPU none repetition; that was
the fastest none repetition, and its median comes from an uncontaminated
run. The eight-GPU comparisons ran without concurrent validation work.

Fixed random requests: 256 input tokens, 128 output tokens, seed 42,
unlimited offered request rate, eight request waves, and eight warmup
requests before each of three repetitions. Prefix cache is flushed after
warmup by the benchmark client. Each repetition completes 128, 512, or
1024 requests for concurrency 16, 64, or 128, with exactly the requested
output token count. Throughput includes HTTP serving, scheduling, prefill,
and decode. The primary metric is generated output tokens/second; total
input-plus-output throughput is exactly three times this value here.

Median of all three repetitions, output tokens/s:

| GPUs | Concurrency | none | DeepEP | DeepEP change |
| --- | ---: | ---: | ---: | ---: |
| 2 | 16 | 2,132.9 | 2,036.6 | -4.5% |
| 2 | 64 | 4,344.2 | 4,338.0 | -0.1% |
| 2 | 128 | 5,496.6 | 5,650.1 | +2.8% |
| 8 | 128 | 11,887.2 | 14,227.0 | +19.7% |

All repetitions, in execution order (output tokens/s):

| GPUs | Concurrency | Backend | Run 1 | Run 2 | Run 3 |
| --- | ---: | --- | ---: | ---: | ---: |
| 2 | 16 | none | 1,999.3 | 2,132.9 | 2,135.2 |
| 2 | 16 | deepep | 1,892.7 | 2,036.6 | 2,040.2 |
| 2 | 64 | none | 4,578.7 | 4,344.2 | 4,335.7 |
| 2 | 64 | deepep | 4,608.4 | 4,325.0 | 4,338.0 |
| 2 | 128 | none | 5,485.0 | 5,496.6 | 5,507.9 |
| 2 | 128 | deepep | 5,650.1 | 5,648.0 | 5,655.9 |
| 8 | 128 | none | 6,840.5 | 11,887.2 | 12,061.4 |
| 8 | 128 | deepep | 7,478.2 | 14,227.0 | 14,293.4 |

The first eight-GPU run is much slower for both backends, so the range must
not be hidden behind the median. The two later eight-GPU runs show the same
throughput advantage. These measurements do not establish a universal win
at small batch sizes or on every EP configuration.

Reproduce with the checked-in `benchmark/bench_marlin_deepep.py`. The exact
model snapshot path used here is assigned below. Obtain that revision with
`huggingface_hub.snapshot_download` if it is not already cached.

```bash
MARLIN_MODEL=/root/.cache/huggingface/hub/models--QuantTrio--Qwen3-Coder-30B-A3B-Instruct-AWQ/snapshots/c58857a7f41c0920f73d1b56678640f9c02017d7
PYTHONPATH=python /opt/sglang/bin/python benchmark/bench_marlin_deepep.py --model "$MARLIN_MODEL" --backends none --dp-attention --capacity 64 --results /tmp/marlin-throughput/none-final
PYTHONPATH=python /opt/sglang/bin/python benchmark/bench_marlin_deepep.py --model "$MARLIN_MODEL" --gpus 0,1,2,3,4,5,6,7 --dp-attention --capacity 16 --concurrency 128 --results /tmp/marlin-throughput/eight-gpu
PYTHONPATH=python /opt/sglang/bin/python benchmark/bench_marlin_deepep.py --model "$MARLIN_MODEL" --backends deepep --dp-attention --capacity 64 --results /tmp/marlin-throughput/deepep-valid-only
```

Use new result directories when rerunning; the script refuses to overwrite
benchmark JSON. It starts and stops the HTTP servers itself. Defaults used
above are concurrency 16/64/128, eight waves, three repetitions, and lengths
256/128. The eight-GPU command explicitly selects concurrency 128.

## Profiling and fixes

Initial TP=EP=2 runs with TP attention and capacity 128 showed DeepEP behind
none at every tested concurrency. Profiling identified unnecessary padded
expert computation, buffer initialization, single-expert reduction, and
invalid-block scans. The implementation now compacts valid expert segments,
builds their block schedule directly from GPU counts, selects tiles from
expected valid rows, and restores only valid output rows. NaN-poisoned
padding tests verify the DeepEP handle's valid-row contract.

The actual baseline also exposed standard Marlin EP passing non-local `-1`
routes to a non-EP kernel during graph capture. Both measured backends use
the corrected common runner; the baseline is not allowed to crash or skip
valid work. Dedicated one-token and multi-token tests cover this fix.

Three separate greedy sanity prompts (64 generated tokens each) produced
matching text for none and DeepEP in the two-GPU server checks. This is a
limited sanity check, not a language-quality evaluation or bitwise logit
parity claim; selected-token log probabilities differed by up to 0.322.

## Boundaries

Results apply to this pinned AWQ model, fixed request lengths, concurrency,
and single-host H200 setup. The earlier component measurements use synthetic
weights and the original implementation, and must not be treated as current
production throughput. Language-quality benchmarks, other pretrained weight
formats, and multi-node performance remain unmeasured.
