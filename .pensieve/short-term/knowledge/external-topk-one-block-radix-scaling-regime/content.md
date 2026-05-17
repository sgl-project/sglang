---
id: external-topk-one-block-radix-scaling-regime
type: knowledge
title: External top-k radix kernels — one-block-per-batch is fast at short ctx, slow at long ctx
status: active
created: 2026-05-17
updated: 2026-05-17
tags: [topk, radix, sparse-attention, gpu-utilization, ftka, raft, double-sparsity]
---

# External top-k radix kernels — one-block-per-batch is fast at short ctx, slow at long ctx

## Source
- Decision: `[[decisions/2026-05-17-ftka-evaluation-scoped-to-selector-only]]`
- Measurement: `benchmark/double_sparsity/repro_session/ftka_comparison/REPORT.md` (§"P3 — full grid, by context", §"Why RAFT loses at 128K")
- Reference kernel: `tsinghua-ideal/flash-topk-attention@d8803b29` →
  `csrc/include/raft_topk.cuh::radix_topk_one_block_kernel` (BitsPerPass=8, BlockSize=512)
- Counter-example: `torch.topk` → CUB many-block radix (fixed wave covering many SMs)

## Summary

Two top-k radix implementations look identical from the outside —
both pick top-k from `[B, L]` scores — but have **opposite scaling
profiles**:

- **One-block-per-batch radix** (RAFT `decode_select_k`,
  some CUB segmented sorts): launches **B** CUDA blocks. Each block
  walks its row sequentially in `num_passes` radix passes. Time
  scales ~**linearly with L** for fixed B; SM occupancy is bounded
  by B regardless of L.
- **Many-block radix** (torch.topk, well-tuned CUB device-wide radix):
  launches enough blocks to cover SM count regardless of B. Time is
  dominated by global memory bandwidth and stays roughly flat with B.

The crossover where one-block becomes worse is around **L = 64K** on
H200 (sm_90, 144 SMs) for `bs ≤ 32`. For DS at production ctx = 128K,
the one-block kernel is **0.47×–0.83× of torch.topk** (i.e., slower).
For ctx = 32K, it's **2× faster** (because torch's launch+wave
overhead dominates the actual work at that size).

## Content

### State transition

```
[external sparse-attn kernel offers top_k op]
        │  measure at the workload's L (seq_len / max_ctx)
        ▼
[is the kernel's parallel dim B (batches) or L (positions)?]
        │
        ├── parallel over L (many blocks per batch, covers SMs)
        │     ──► time ≈ flat in L for memory-bound; flat in B too
        │     ──► CANDIDATE: time it across the production grid
        │
        └── parallel over B (one block per batch row)
                │  measure time vs L for fixed B
                │
                ├── time ≈ flat in L → unlikely, but candidate
                │
                └── time scales ~linear in L
                        │
                        ├── num_blocks (= B) >> num_SMs?
                        │       │
                        │       ├── yes — competitive; SMs saturate
                        │       │
                        │       └── no — UNDER-UTILIZED; reject
                        │             for long-L production targets
                        │
                        └── unsure → measure at production ctx (128K
                              for DS), do NOT extrapolate from short-L
                              microbenches in upstream READMEs
```

### Symptom → root cause → location

| Symptom (during evaluation) | Root cause | Where to confirm |
|----|----|----|
| External kernel benchmark shows N× faster than baseline at short ctx (32K) but slower at long ctx (128K) | One-block-per-batch radix: B blocks << num_SMs, time ~linear in L | `decode_select_k` block-grid shape; `[[knowledge/ds-external-sparse-kernel-layout-mismatch/content]]` for the analogous score-kernel trap |
| Speedup looks great in the kernel author's benchmark, vanishes in your microbench | Author benchmarked at small L or large B; production has the opposite | Their `benchmark/` scripts vs `microbench_ftka_backends.py` shape grid |
| RAFT-style top-k claims O(L) per-row but pays a 4×-passes constant for fp32 | `calc_num_passes<fp32, BitsPerPass=8>() == 4`; each pass scans L | `csrc/include/raft_topk.cuh::radix_topk_one_block_kernel`, the `for (pass = 0; pass < num_passes; ++pass)` loop |
| At very small `bs` (1–4) the external kernel still wins at long ctx | This is the regime where torch.topk's CUB radix can't amortize its launch wave either | Both lose to each other across `bs × L`; pick by where production sits |

### Boundaries and ownership

- **Within scope**: any sparse-attention library that offers a
  drop-in top-k routine over `[B, L]` scores — RAFT's
  `decode_select_k`, future FTKA / FlashInfer / vLLM radix top-k
  ports, sgl-kernel `fast_topk_transform_fused`.
- **Out of scope**: top-k variants where the parallelism is over L by
  design (e.g., segmented scan + reduce); those scale flat in L.

### Anti-patterns

- **"Their README shows 2× faster, integrate it."** Their README likely
  benchmarked at the wrong `(B, L)` for your workload. Measure at
  *your* production shape — DS at ctx ∈ {64K, 128K}, bs ∈ {16, 32}.
- **"It passes parity, it's CUDA-graph safe, ship it."** Parity and
  capture-safety are necessary, not sufficient. The third leg is
  measured speedup at the production envelope.
- **"It's slower by 5% — close enough."** No: the production envelope
  here is 21–37% slower at every headline shape. There is no path
  from there to acceptable.
- **"Build their full library against my torch/CUDA stack."** External
  sparse-attn kernel libraries authored against older flashinfer pin
  fast (FTKA d8803b29 pins flashinfer 0.2.0.post1 / torch 2.5 /
  numpy 2.1.2 in its `requirements.txt`; SGLang ships flashinfer
  0.6.11 / torch 2.11). Build-glue drift (vec_t<uint8_t>
  redefinition, nvcc 13 strict-mode half-init, JIT API moves) is the
  norm. Carve a minimal TU instead of porting the whole library.

### Verification signals

After microbenching an external top-k candidate, the integration is
**measurably safe** if all of:

1. `parity_match = ok` against torch.topk on all shapes.
2. `graph_status = ok` under an isolated `torch.cuda.graph` probe.
3. **Median speedup ≥ 1.15× across the production envelope** (`bs ≥ 16,
   ctx ≥ 64K, top_k ≥ 1024` for DS). Per-shape, NOT global-grid
   median — short-L wins must not be allowed to mask long-L losses.

The integration is **structurally unsafe at long L** if:

- The kernel launches `O(B)` blocks total (one per batch row).
- Measured time scales ~linearly with `L` at fixed `B`.
- At your production `B`, you have `B < num_SMs` (true for DS at
  `h_kv=1, bs ∈ {1, 4, 8, 16, 32}` on a 144-SM H200).

## When to Use

Read this when:

- Considering ANY external top-k kernel (RAFT, CUB segmented, custom
  CUDA) as a substitute for `torch.topk` in a sparse-decode hot path.
- Reviewing a benchmark claim of "Nx faster than torch.topk" — verify
  it was measured at *your* `L` and `B`, not the author's defaults.
- Designing a new sparse-decode top-k phase where the kernel's
  parallel-dim shape matters more than its theoretical complexity.

## Context Links
- Based on: [[knowledge/ds-external-sparse-kernel-layout-mismatch/content]]
- Leads to: [[decisions/2026-05-17-ftka-evaluation-scoped-to-selector-only]]
- Related: [[knowledge/ds-flashinfer-top-k-page-table-boundaries/content]]
