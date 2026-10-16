# Muse Spark on MI355X: decode fusion passes (2026-09-07)

Stack: `rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260905`, sglang dc28438 (+ patches/03 Plan B), aiter c16d44b9
(+ patches/01), overlay = this repo (branch `rocm/decode-fusion`), TP8/EP8, `ATTN=aiter PAGE_SIZE=16`.
Regression gate: gsm8k 1319q 5-shot (Accuracy 0.955-0.958 / Invalid 0 / 0 GPU faults on every run).
Perf gate: fixed-length bench (`bench_e2e.py`, 256 tokens, ignore_eos) and per-bs `gen throughput` in the
server log. gsm8k wall time is dominated by the longest greedy generations and is NOT a perf metric.

## Why: decode was kernel-count bound
Eager trace (BF16, conc 1): ~2,630 kernels per decode step (46.7/layer, ~8 us each); hipgraph replay wall
(24.2 ms) ~= eager GPU busy (21 ms), i.e. the step is the sum of thousands of tiny kernels.
Hot spots: RMSNorm decomposed into pow/mean/rsqrt/mul on the fp32 stream + `weight+gain` add per call,
gated residual in eager, hipBLASLt skinny GEMMs with split-K reduce kernels, MoE glue, qk-norm + casts.

## Passes (MXFP4 numbers, aiter prefill / Triton or 3D decode)
| pass | change | conc1 step | fixed-256 bench conc 1/8/32/64/128 (tok/s) |
|---|---|---|---|
| baseline | aligned stack, unfused | 20.9 ms | - (gsm8k 1280 tok/s) |
| 1 | cached effective norm weights; cross-layer fused FFN boundary (`FusedPostNormResidualPreNorm` with next layer's pre_attn_norm / final norm) | 14.5 | - |
| 2 | fused q/k RMSNorm(+scale) kernel; MXFP4 mid/large MoE tiles (16/64/256 G4, 64/64/128 G8 w4) | 14.0 | - |
| 3 | MoE split-K=4 (<=32 slots) + one-program alignment (<=64 slots) + grid bound; BF16 small tile K128; mid/large crossover rows<=6144 (MXFP4) / 2048 (BF16) | 13.3 | 83/523/1654/2776/4045 |
| 4a | fp32 router as split-K Triton GEMV (M<=64) | 12.85 | 86/539/1704/2820/4035 |
| 4b | pooler/unpooler via aiter `tgemm.mm` at decode-sized M | 12.2 | 90/559/1750/2939/4043 |
| +Plan B | decode on aiter unified-attention 3D (patches/03) instead of Triton decode | 11.8 | 92/566/1782/2993/4123 |
BF16 pass 3: conc1 24.2 -> 14.6 ms, bench 75/378/1017/1659/2529 (weight-bandwidth bound at high conc).
IL8192/OL128 conc4 (MXFP4, fused): Plan A TTFT 421 ms / TPOT 22.8 ms / 148.6 tok/s; Plan B 403 / 22.3 / 154.2.
Kernels per layer: 46.7 -> ~24.

## Switches (ROCm only; default on unless noted)
`MUSE_FUSE_FFN_BOUNDARY`, `MUSE_FUSE_QK_NORM`, `MUSE_ROUTER_GEMV`, `MUSE_AITER_DENSE` (default off: slower than
hipBLASLt on the pinned aiter), `MUSE_MOE_SPLITK` (4),
`MUSE_MOE_ALIGN` (sgl|torch), `MUSE_MOE_FP4_MODE` (dot_scaled|manual), `MUSE_MOE_BACKEND` (triton|native|flydsl).
`flydsl` needs `patches/04-aiter-muse-moe-flydsl.patch` on aiter `c16d44b9`.

## Caveats
- Norm effective weights and fused-op weight caches are computed in `post_load_weights`; a later weight
  reload must call it again.
- Eager micro-benchmarks of small kernels are CPU-launch bound on ROCm (~15 us/launch): time with CUDA-graph
  replay. torch profiler cannot see hipgraph kernels; rocprofv3 wrapping the server aborts the HSA queue.
- Standalone hipBLASLt shows algo-selection pathologies (e.g. 557 us at M=32) not seen inside the server.

## Remaining per-layer budget at bs1 (~190 us): dense GEMMs ~65, MoE ~42, all-reduce ~30, attention ~20,
router ~12, norms/glue ~20. Next: aiter tuner entries for qkv/pooler/unpooler shapes (asm), aiter MoE stage2
without top-k combine (needs aiter change), Plan C FlyDSL SWA decode, all-reduce+norm fusion.
