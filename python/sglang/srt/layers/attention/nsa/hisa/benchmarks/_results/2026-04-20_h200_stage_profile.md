# profile_stages.py — 2026-04-20 (H200, exclusive GPU)

Per-stage CUDA-event timing (includes Python dispatch). 5 warmup + 20 iters,
L2 flushed per iter. block config: k_block_size=128, block_topk=64.
Companion to `2026-04-20_h200.md` (end-to-end benchmark).

## Decode (paged, next_n=1, paged_block_size=64)

|   B |   ctx | mean_pool     | pool_mqa      | topk+cast     | sparse_mqa    | TOTAL |
| --: | ----: | ------------- | ------------- | ------------- | ------------- | ----: |
|   1 |  4096 | 0.016 (11.5%) | 0.007 ( 5.0%) | 0.009 ( 6.3%) | 0.106 (77.1%) | 0.138 |
|   1 | 16384 | 0.016 ( 6.7%) | 0.008 ( 3.3%) | 0.013 ( 5.5%) | 0.204 (84.6%) | 0.241 |
|   1 | 65536 | 0.022 ( 8.5%) | 0.012 ( 4.6%) | 0.016 ( 6.2%) | 0.210 (80.7%) | 0.260 |
|   8 |  4096 | 0.017 (12.5%) | 0.007 ( 5.2%) | 0.009 ( 6.4%) | 0.103 (75.8%) | 0.135 |
|   8 | 16384 | 0.027 (10.9%) | 0.008 ( 3.2%) | 0.013 ( 5.3%) | 0.199 (80.6%) | 0.247 |
|   8 | 65536 | 0.065 (21.5%) | 0.013 ( 4.2%) | 0.016 ( 5.4%) | 0.208 (68.8%) | 0.303 |
|  32 |  4096 | 0.027 (17.4%) | 0.007 ( 4.7%) | 0.013 ( 8.3%) | 0.108 (69.6%) | 0.155 |
|  32 | 16384 | 0.065 (21.7%) | 0.008 ( 2.8%) | 0.013 ( 4.4%) | 0.214 (71.1%) | 0.300 |
|  32 | 65536 | 0.205 (44.6%) | 0.013 ( 2.9%) | 0.017 ( 3.6%) | 0.224 (48.9%) | 0.459 |
|  64 |  4096 | 0.041 (24.1%) | 0.008 ( 4.4%) | 0.013 ( 7.7%) | 0.108 (63.8%) | 0.169 |
|  64 | 16384 | 0.112 (32.5%) | 0.009 ( 2.5%) | 0.013 ( 3.9%) | 0.211 (61.2%) | 0.345 |
|  64 | 65536 | **0.391 (61.0%)** | 0.014 ( 2.1%) | 0.017 ( 2.6%) | 0.220 (34.3%) | 0.641 |

## Prefill (single-seq causal)

| seq_len | mean_pool    | pool_mqa      | topk+cast     | sparse_mqa     | TOTAL  |
| ------: | ------------ | ------------- | ------------- | -------------- | -----: |
|    4096 | 0.009 ( 1.1%) | 0.085 (11.2%) | 0.019 ( 2.5%) |  0.646 (85.1%) |  0.759 |
|    8192 | 0.009 ( 0.3%) | 0.157 ( 5.9%) | 0.050 ( 1.9%) |  2.442 (91.9%) |  2.659 |
|   16384 | 0.009 ( 0.2%) | 0.300 ( 5.7%) | 0.137 ( 2.6%) |  4.813 (91.5%) |  5.259 |
|   32768 | 0.010 ( 0.1%) | 0.593 ( 5.6%) | 0.457 ( 4.3%) |  9.598 (90.0%) | 10.658 |
|   65536 | 0.015 ( 0.1%) | 1.480 ( 6.6%) | 1.582 ( 7.1%) | 19.228 (86.2%) | 22.305 |

## Interpretation

**Decode: mean_pool dominates at large B·ctx.**
- mean_pool cost scales with total KV tokens (~B · ctx). At (B=64, ctx=65536)
  it is 0.391 ms — **61% of total**, nearly 2× the sparse_mqa stage.
- Baseline (from `2026-04-20_h200.md`): B=64 ctx=65536 = 0.323 ms.
  hisa total = 0.641 ms, of which 0.391 ms is mean_pool. Without mean_pool,
  remaining hisa stages (0.250 ms) would beat baseline.
- This is the motivation for the phase-2 pool K cache:
  mean_pool becomes incremental (per new KV token) instead of full-recompute
  every decode step.

**Prefill: sparse_mqa dominates everywhere (85-92%).**
- mean_pool is ≤ 1.1% of total at every seq_len — already negligible.
- Future optimization leverage is on sparse_mqa, not mean_pool.
- topk cost grows faster than linear with seq_len (2.5% at 4K → 7.1% at 64K)
  but is still under 10% — not a current bottleneck.

## Command

```
python python/sglang/srt/layers/attention/nsa/hisa/benchmark/profile_stages.py \
    --block-configs 128:64 \
    --seq-lens 4096 8192 16384 32768 65536 \
    --batch-sizes 1 8 32 64 \
    --context-lens 4096 16384 65536
```
