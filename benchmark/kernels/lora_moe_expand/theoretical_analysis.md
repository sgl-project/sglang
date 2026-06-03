# Theoretical analysis: is 20 µs reasonable for the Kimi-K2.5 down-proj expand-add?

Companion to `journal_tom.md`. The sweep's best result for the Kimi-K2.5 down-proj
LoRA-B expand-add is **20.12 µs** (`block_m=16 block_n=512 group_m=1 warps=4`, 1× GB200,
bs=64, r=16). Judged as a GEMM by FLOPs that looks absurdly slow. This note works the
roofline and concludes it is **reasonable**: the op is deeply memory-bound on weight
streaming, and 20 µs is ~40% of HBM peak — roughly 2–2.5× off the theoretical floor,
which is typical for a tiny-K grouped GEMM.

## Problem shape

This is not a dense GEMM. It is a grouped (MoE) GEMM with a tiny reduction dim, followed
by a routed atomic-add reduction:

| symbol | meaning | value |
|---|---|---|
| `M` | token-expert rows = `bs · top_k` | 64 · 8 = **512** |
| `K` | reduction dim = LoRA rank `r` | **16** |
| `N` | down-proj output hidden | **7168** |
| `E` | routed experts (TP8, no-EP) | **384** |
| `E_hit` | distinct experts actually touched | **~284** (see below) |
| `out` | fused per-token output `[bs, N]` | `[64, 7168]` |

`E_hit`: each token picks `top_k=8` distinct experts (`randperm`), so a given expert is
missed by one token with prob `(1 - 8/384)`, and over 64 tokens
`E_hit ≈ 384·(1 - (1 - 8/384)^64) ≈ 284`. The grouped GEMM iterates `expert_ids` and
skips empty experts, so only ~284 of 384 weight matrices are ever read (seed-dependent;
upper bound 384).

## 1. Compute roofline — not the bound

```
FLOPs = 2 · M · K · N = 2 · 512 · 16 · 7168 ≈ 117.4 MFLOP
20.12 µs → 117.4e6 / 20.12e-6 ≈ 5.84 TFLOP/s
```

B200 bf16 dense peak ≈ 2250 TFLOP/s → **0.26% utilization**. If you stopped here you'd
call the kernel broken. But the FLOPs are irrelevant here — see the arithmetic intensity
below.

## 2. Memory roofline — the real bound

The dominant traffic is streaming each hit expert's LoRA-B weight `[N, r]` bf16 exactly
once. Padding on `M` (block_m=16 vs avg `M_e ≈ 1.8` rows/expert) wastes compute but reads
no extra weight bytes: per `(expert, n_tile)` the B tile is read once regardless of valid
M rows.

```
weight bytes  = E_hit · N · K · 2
              = 284 · 7168 · 16 · 2 ≈ 65.1 MB   (≈ 62 MiB)
output bytes  = memset [64,7168] bf16 (0.92 MB) + atomic writeback (~0.92 MB) ≈ 1.8 MB
input bytes   = intermediate [512,16] bf16 ≈ 16 KB  (negligible)
total         ≈ 67 MB,  ~97% of it weights
```

Arithmetic intensity:

```
AI = 117.4 MFLOP / 67 MB ≈ 1.75 FLOP/byte
ridge point (B200) = 2250 TFLOP/s ÷ 8 TB/s ≈ 281 FLOP/byte
```

`AI ≈ 1.75 ≪ 281` → memory-bound by ~160×. This is a weight-streaming op, full stop.

### Floor vs achieved

| quantity | value |
|---|---|
| HBM bandwidth (B200, HBM3e) | ~8 TB/s |
| roofline floor = 67 MB / 8 TB/s | **~8.4 µs** |
| achieved | **20.12 µs** |
| effective bandwidth = 67 MB / 20.12 µs | **~3.3 TB/s** |
| % of HBM peak | **~42%** |

So the kernel runs at ~42% of HBM peak, ~2.4× off the streaming floor.

## 3. Why ~42% and not ~100%

The weight read (65 MB) is already optimally amortized — each hit expert is read once,
shared across its `M_e` tokens — so the gap is per-tile efficiency, not redundant traffic:

- **K=16 is a single MMA step.** There is no K-loop to amortize the tile prologue/epilogue
  over. Each `[block_m,16]·[16,block_n]` tile does minimal math between its load and its
  atomic-add epilogue, so memory latency is poorly hidden.
- **Atomic-add reduction contention.** With `fuse_sum_all_reduce` + `mul_routed_weight`,
  all 512 per-expert deltas atomic-add into just 64 output rows. Many tiles contend on the
  same `[N]` row → serialization in the epilogue.
- **~3976 tiny tiles.** `E_hit · cdiv(N, block_n) = 284 · 14 ≈ 3976` blocks, each with
  `M_e ≤ 8 < block_m=16` (≈9× row padding). Lots of small grids → launch/scheduling and
  tail/wave-quantization overhead relative to the trivial per-tile work.

These also explain the sweep: `block_m=16` wins because larger block_m is pure M-padding
waste; `block_n=512` wins because a wider N tile amortizes the fixed per-tile overhead and
improves coalescing of the long `N=7168` weight rows.

## Conclusion

**20 µs is reasonable and in the right order of magnitude.** The op is memory-bound on
~65 MB of weight streaming with a hard floor of ~8 µs; at ~42% of HBM peak the kernel is
~2.4× off that floor, which is normal for a grouped GEMM with `K=16` and ~2-row experts.

There is plausibly ~1.5–2× headroom (toward ~10–13 µs) for a kernel that hides the K=16
latency better and reduces atomic contention, but the FLOPs-based "0.26% of peak" framing
is misleading — there is no compute headroom to chase, only bandwidth efficiency.

### Caveats / sensitivities
- HBM peak taken as ~8 TB/s for B200; the % numbers scale inversely if the true sustained
  figure differs.
- `E_hit ≈ 284` is the expected value for this routing; the actual seed (`manual_seed(0)`)
  fixes it. If the kernel instead read all 384 experts the floor would be ~10.5 µs and the
  achieved efficiency ~56% of peak.
- Output atomic traffic is treated as ~2 MB (target `[64,7168]` largely L2-resident); only
  weights set the floor.
