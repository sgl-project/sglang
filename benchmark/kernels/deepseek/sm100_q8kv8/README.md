# DeepSeek-V4-Flash SM100 Q8KV8 experiments

This directory contains reproducible B200 experiments for comparing the full
DeepSeek-V4-Flash sparse-prefill backend functions:

- BF16 golden: `DeepseekV4AttnBackend._forward_prefill_sparse`
- SM100 Q8KV8: `DeepseekV4AttnBackend._forward_prefill_sparse_q8kv8`

The benchmark includes Q conversion, packed-KV gather/conversion, sparse index
adaptation, attention, and output handling. It reports C0/C4/C128 separately
and the model's 43-layer weighted estimate (`C0x3 + C4x20 + C128x20`).
The default uses the production TP8 shape with 8 local Q heads; pass
`--num-heads 64` to measure the TP1/full-head kernel shape separately.
The default sequence-length matrix is 512, 1024, 2048, 4096, 8192, 16384,
and 32768 tokens.

Run the long-sequence accuracy gate on the assigned physical GPU 7 only. By
default it compares 12 cases (4K/8K/16K/32K x C0/C4/C128) against the complete
BF16 sparse-prefill backend and writes a timestamped JSON artifact:

```bash
CUDA_VISIBLE_DEVICES=7 python benchmark/kernels/deepseek/sm100_q8kv8/validate_dsv4_q8kv8_accuracy.py
```

The default acceptance thresholds are cosine >= 0.999, mean absolute error <=
0.001, and maximum absolute error <= 0.02. The script exits nonzero if any
case fails.

Run the performance matrix on the same physical GPU:

```bash
CUDA_VISIBLE_DEVICES=7 python benchmark/kernels/deepseek/sm100_q8kv8/benchmark_dsv4_q8kv8.py
```

Timestamped CSV and JSON artifacts are written to `results/`. The script has a
hard safety guard and refuses to run if any physical GPU other than 7 is made
visible.

The 12-case long-sequence accuracy gate passed on B200 GPU7 with cosine
similarity in `[0.99992138, 0.99993408]`, mean absolute error no greater than
`4.23e-5`, and maximum absolute error no greater than `0.00342`. The artifact is
`results/20260827-072748-gpu7-v49-long-accuracy.json`.

The repository-integrated 30-repeat performance regression is
`results/20260827-072925-gpu7-v49-integrated-long-performance.json`. Its
DeepSeek-V4-Flash weighted BF16/Q8 speedups are 1.395x, 1.324x, 1.318x, and
1.321x at 4K, 8K, 16K, and 32K respectively.

## Current B200 result

The current best valid CUDA configuration is native E4M3 tcgen05 with an
internal 32-head tile and a 128-key double-block tile. D512 uses three KV
stages; D576 uses two to remain within the SM100 shared-memory limit. The KV
producer dispatch is adaptive: 384 threads / 7 producer warps below 4096 query
tokens and 512 threads / 11 producer warps at or above 4096. It also retains
compact TP8 Q storage, four-warp TMEM output extraction, direct active-head
global stores, KV `EVICT_LAST`, and the backend-only no-metadata
specialization. The stable 30-repeat GPU7 result is
`results/20260827-065905-gpu7-best-valid-v49-adaptive-double-block-full.json`:

| Sequence | BF16/Q8 weighted speedup | Gain |
| ---: | ---: | ---: |
| 512 | 0.873x | -12.69% |
| 1024 | 0.875x | -12.49% |
| 2048 | 1.211x | +21.12% |
| 4096 | 1.397x | +39.70% |
| 8192 | 1.333x | +33.29% |
| 16384 | 1.317x | +31.75% |
| 32768 | 1.320x | +32.05% |

The requested 1.5x target is not yet met. Results marked as `smoke` or from
negative experiments must not be used as the best result. Results through v33
are also invalid as correctness baselines: they read only warp0/warp2's TMEM
output fragments and duplicated/permuted the remaining 256 columns. Rejected
performance variants include two/four KV buffers, N=128 with two buffers,
cp.async KV loading, per-K-tile barriers, 224/256-thread role layouts, explicit
TMEM offsets, direct in-CTA BF16-Q conversion, dual-block softmax batching, and
M=16 tcgen05 (unsupported by both SM100 WS and non-WS 1SM TMEM traits). The
occupancy-forced two-buffer variant is also rejected:
`launch_bounds(..., 2)` reduced the trusted kernel from 147 to 86 registers but
introduced a 176-byte stack spill and regressed 32768/C4 from 3.626 ms to
4.696 ms. An N=128/two-buffer bring-up was also slower because 256 tokens of
lookahead did not hide the larger random-TMA stage; the production D512
three-buffer version increases lookahead to 384 tokens. The next large
opportunity is removing or fusing the Q cast and combined-index adapter work
before the attention kernel.
