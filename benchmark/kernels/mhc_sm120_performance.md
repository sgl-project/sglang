# SM120 mHC performance validation

The revised tuning speeds up the fused stats and Sinkhorn path by 1.46–1.57x for the tested M values 4096, 8192 and 16384 on RTX PRO 5000. It preserves the non-fused and compensated bf16x3 paths at BLOCK_K=64. At M=1/8 the fused call takes approximately 2–4 microseconds longer than the BK64 baseline; this tuning does not make every batch size faster.

## Code and environment

- Validated implementation: `cc20e6829d5ab08a0e25ba93a23e9c9f3d4c169e`, following original PR head `ac16bd53583cf744812d63c516453414160492cf`.
- Candidate mHC module SHA256: `5d1be6af017c25d4440a5e17c831c287a44af5d50a3f0c7ea8284657ca6e8b29`.
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, compute capability 12.0. Microbenchmarks use physical GPU 2.
- Driver: 580.126.09; Torch 2.14.1+cu130; CUDA 13.0; Triton 3.8.0. Installed SGLang distribution metadata: 0.5.23.post1; the imported Python source comes from the isolated PR checkout.
- GPU clocks and power limits use the existing automatic settings. Measurements were recorded on 2026-11-09.

## Revised dispatch

`hc_mix_stats_sinkhorn` queries the input device when first called and caches the result per device. Exact SM120 selects BK32; SM121, datacenter NVIDIA GPUs and HIP retain BK64. SM120 BK32 uses row tiles of at least 16. The non-fused `hc_mix_stats` stays at BK64, and `hc_mix_stats_sinkhorn_bf16x3` has its own BK64 constant for both the shape assertion and launch.

The minimum 16-row tile is necessary for the arithmetic contract. Before that fix, processing all 4096 reference rows in batches of eight at seed 7 changed 125/84/223 rows of pre/post/comb respectively. The full-row regression now passes across seeds 7, 11 and 19 and K values 2560 and 20480. Restoring the original eight-row policy makes that regression fail.

BLOCK_K can also affect split-K partitioning: at K=20480 both widths select 80 slices, while at K=2560 BK64 selects 40 and BK32 selects 80. The slice choice remains independent of batch size.

## Kernel timing protocol

The measured entries are `hc_mix_stats`, `hc_mix_stats_sinkhorn` and `hc_mix_stats_sinkhorn_bf16x3`. All three tile policies run in the same source module: BK64 everywhere, the original PR tile policy, and the revised production dispatch. This isolates the tile choices rather than comparing different package versions.

K=20480, MIX=24, hc_mult=4, bf16 activations, fp32 weights scaled by 1/sqrt(K), rms_eps=1e-20, hc_eps=1e-6 and 20 Sinkhorn iterations. The partial kernel uses four warps, one stage and tf32x3. The bf16x3 kernel retains four warps and three stages. bf16x3 cases use M ≥ 4096.

Each graph captures one wrapper call and external CUDA timing events. The cold condition zeroes a 256 MiB buffer before the start event; its cost is excluded. The warm condition reuses the input and weights. Compilation and initial warmup are excluded. Each cell pools five trials of 31 samples, with policy order shuffled within trials, and reports the median. Cold and warm describe this cache protocol, not measured hardware cache-miss rates. These are GPU execution times; Python wrapper and server costs are measured separately by the serving experiment.

## Fused stats and Sinkhorn

| M | Cold BK64 us | Cold revised us | Speedup | Warm BK64 us | Warm revised us | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.38 | 20.48 | 0.800x | 9.89 | 11.52 | 0.858x |
| 8 | 18.43 | 21.92 | 0.841x | 11.71 | 13.66 | 0.857x |
| 256 | 59.39 | 57.34 | 1.036x | 38.18 | 29.70 | 1.286x |
| 1024 | 172.03 | 154.94 | 1.110x | 127.39 | 91.58 | 1.391x |
| 4096 | 705.95 | 484.77 | 1.456x | 691.71 | 460.42 | 1.502x |
| 8192 | 1398.78 | 931.84 | 1.501x | 1419.97 | 916.19 | 1.550x |
| 16384 | 2853.89 | 1845.25 | 1.547x | 2875.94 | 1834.50 | 1.568x |

The gain at M=256/1024 depends strongly on cache reuse. Small-M results are a regression against BK64 under both cache conditions; the previous claim that the optimization is never slower is removed.

## Compensated bf16x3

| M | Cache | BK64 baseline us | Original tile policy us | Revised us | Original time / baseline |
| --- | --- | --- | --- | --- | --- |
| 4096 | cold | 192.51 | 284.67 | 192.51 | 1.479x |
| 4096 | warm | 174.30 | 259.07 | 174.14 | 1.486x |
| 8192 | cold | 368.64 | 546.82 | 370.69 | 1.483x |
| 8192 | warm | 341.47 | 508.16 | 340.32 | 1.488x |
| 16384 | cold | 706.56 | 1056.77 | 704.51 | 1.496x |
| 16384 | warm | 660.70 | 1008.74 | 660.58 | 1.527x |

The shared BK32 policy increased bf16x3 time by approximately 1.48–1.53x on these cases. The independent BK64 constant keeps the revised path within measurement variation of baseline and gives identical outputs. The non-fused stats path also produces identical baseline outputs and unchanged timings in the raw matrix.

## Correctness and regression checks

All 11 SM120 GPU tests pass. Coverage includes exact device dispatch and caching, module-body capability queries, bitwise row consistency across M=1/8/9/16/32/2048/2049/4096 and every reference row routed through a batch of eight, BK64 numerical comparison, and bf16x3 independence from the partial width. The regression discrimination check rejects the original eight-row tile policy.

Across the recorded fused microbenchmarks, maximum absolute output difference from BK64 is 3.576e-07; maximum relative L2 error is 9.650e-08. Non-fused stats and revised bf16x3 have zero differing elements against baseline. The fused accuracy test uses rtol=1e-5 and atol=2e-6.

The module-body import test preloads dependencies and then rejects eager capability queries during reload. Existing Sail platform discovery independently initializes CUDA while importing dependencies; this patch removes the mHC tuning's added eager query without claiming to fix all project-wide import effects.

Syntax, Ruff, isort and whitespace checks pass.

## Reproduce the kernel experiment

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python \
  benchmark/kernels/bench_mhc_sm120.py \
  --output results/kernel.json --trials 5 --repetitions 31

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/layernorm/test_mhc_sm120.py
```

The report does not carry forward the older combined-branch GSM8K scores or the older 2.5 percent prefill claim as measurements of this revised code.


## Serving measurements

DeepSeek-V4.1-Flash-1017-NVFP4 on all eight GPUs, DP8 attention and EP8, DSPARK disabled, CPU Engram, FlashInfer CUTLASS MoE and FP8 GEMM. Context length 65536; CLI chunked prefill 32768 (resolved to 4096 per attention-DP rank), CLI maximum 16 running requests, CLI decode CUDA graph limit 8. The checkpoint shards are staged in shared memory to avoid FUSE mmap loading stalls; both arms use the same copied weights. The baseline and candidate source trees differ only in the fused SM120 tile selection returning 64 or 32. Both use the same minimum-row-tile rule, which is active only at BK32.

2 fresh server boots run in baseline, updated order. Each boot has one warmup and eight measured requests per case. Every request flushes the prefix cache first; all recorded cached-token counts are zero. Inputs are seeded token sequences with exact lengths. TTFT is measured at the client from the first SSE token chunk. TPOT is (completion latency minus TTFT)/(output tokens minus one); this is an average rather than a per-token arrival trace.

| Boot | Policy | 8K/1 TTFT ms | 32K/1 TTFT ms | 3K/128 TTFT ms | 3K/128 TPOT ms |
| --- | --- | --- | --- | --- | --- |
| 0 | baseline | 786.49 | 3093.07 | 332.74 | 26.549 |
| 1 | updated | 769.52 | 3019.92 | 328.08 | 26.549 |

The following comparison uses the per-boot median for each policy. Input throughput gains use the inverse TTFT ratio. This single A/B pair does not estimate variation between fresh server boots.

| Input/output tokens | Baseline TTFT ms | Revised TTFT ms | Observed input throughput change |
| --- | --- | --- | --- |
| 8192/1 | 786.49 | 769.52 | 2.20% |
| 32768/1 | 3093.07 | 3019.92 | 2.42% |
| 3072/128 | 332.74 | 328.08 | 1.42% |

3K/128 median TPOT: 26.549 → 26.549 ms/token (observed medians in this pair).

CPU Engram huge-page coverage at load time differed between boots:

| Policy | Engram layer | Coverage across ranks |
| --- | --- | --- |
| baseline | 1 | 48–85% |
| baseline | 14 | 0% |
| updated | 1 | 100% |
| updated | 14 | 5–32% |

The shared-memory checkpoint occupies about 400 GiB of host RAM. Huge-page allocation was not held fixed, and this single A/B pair cannot separate its effect from the mHC tuning or estimate variation between fresh boots. Treat these serving deltas as observations, not an isolated causal speedup. The GPU-only kernel comparison does not use CPU Engram tables.

## Review follow-up

This report supersedes the original PR description's shared BK32 policy and its claim that all batch sizes improve. In particular, the compensated bf16x3 path stays at BK64, the fused BK32 path uses row tiles of at least 16, and device capability is queried lazily. The serving observations above are reference measurements, not an isolated causal speedup.
