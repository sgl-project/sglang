Status: implemented

# Marlin + DeepEP evidence

TL;DR: Real two-rank H200 execution passes all five quantization families,
including empty ranks and decode graph replay. Three TP=EP=2 server modes pass.

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
- Dispatch adapters: 24 passed, covering active one-token inputs, invalid
  routes, biased experts, supported activations, and graph replay with changing
  valid counts (including all-zero counts).
- Standard Marlin regressions: 6 passed (26 deselected) in 251.55 seconds,
  including NVFP4 with/without bias and large non-EP scheduling.
- Distributed GPTQ4, GPTQ8, AWQ, MXFP4, NVFP4: 40 format/batch/mode
  combinations passed. Twenty low-latency captures each replay twice,
  including uneven batches, rank-local expert skew, one empty rank, and
  both empty ranks. The same AUTO dispatcher switches modes.
- Server normal, low_latency, auto: 3 passed in 148.09 seconds. Each uses
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
absolute/relative error budget. Masked padding must be exactly zero.

## Component latency

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

## Boundaries

These measurements establish execution and numerical correctness on H200,
not production throughput or a speedup over another backend. Weights are
synthetic; the server creates a small local Qwen3 MoE with dummy AWQ weights,
so there is no external checkpoint revision or language-quality result.
Full pretrained checkpoint loading and multi-node execution remain unmeasured.
