---
id: ds-external-sparse-kernel-layout-mismatch
type: knowledge
title: Evaluating external sparse-attention kernels — the paged-cache vs flat-cache trap
status: active
created: 2026-05-17
updated: 2026-05-17
tags: [sparse-attention, ftka, flashinfer, kv-cache, paged-cache, double-sparsity]
---

# Evaluating external sparse-attention kernels — the paged-cache vs flat-cache trap

## Source
- Decision: `.pensieve/short-term/decisions/2026-05-17-ftka-evaluation-scoped-to-selector-only.md`
- Microbench harness: `benchmark/double_sparsity/repro_session/microbench_ftka_backends.py`
- DS K-label cache layout: `python/sglang/srt/layers/attention/triton_ops/double_sparsity_native_decode.py` (`_ds_native_score_kernel`)

## Summary

External sparse-attention kernels published with vLLM/FlashInfer
heritage (FTKA's `batched_sparse_gemv`, FlashInfer's
`top_k_page_table_transform`, etc.) almost always consume a **paged KV
cache** via `kv_indices` / `kv_indptr` / `kv_last_page_len`. DS's
K-label side cache is **flat row-indexed** by `req_to_token[bs,
max_ctx]`. Before benchmarking the external kernel, decide whether
the per-step layout transform fits inside the speedup envelope — if
not, the kernel is structurally adverse, not "almost there."

## Content

### State transition

```
[external kernel published]
        │  inspect signature
        ▼
[is the cache layout flat or paged?]
        │
        ├── flat-row-indexed (matches DS)   ──► evaluate as drop-in: time it, parity-check, capture-probe
        │
        └── paged (kv_indices / kv_indptr / kv_last_page_len)
                 │
                 ├── can we view our flat cache as page_size=1?
                 │       │
                 │       ├── yes — measure: (gemv_us + xform_us) vs our score_us
                 │       │     • amortizable across steps? typically NO (req_to_token changes per step)
                 │       │     • allocation cost under CUDA graph capture? typically PROHIBITIVE
                 │       │
                 │       └── no — REJECT until our cache becomes paged
                 │
                 └── would we paginate DS's K-label cache to match?
                         • cost: rewrite K-label alloc, score kernel, set/append, calibration tooling
                         • benefit: only this one kernel's speedup
                         • verdict: usually out of proportion to gain
```

### Symptom → root cause → location

| Symptom (during evaluation) | Root cause | Where to confirm |
|----|----|----|
| External kernel benchmark shows X% faster than ours **per-call** | only measures the kernel itself, not the per-step transform | `microbench_ftka_backends.py::_FtkaScoreAndSelectRunner.setup` — the `layout_transform_us` field |
| Server crashes mid-init with "OOM" or "cudaMallocAsync failed" when the new kernel is wired | per-step transform allocates `~bs × max_ctx × sizeof(int32)` every decode step | `req_to_token` is updated each step; flattening it is not amortizable |
| CUDA-graph capture fails with "operation not supported" | layout-transform allocator inside captured region | replace the inline transform with a pre-allocated scratch + in-place fill |

### Boundaries and ownership

- **Selector phase**: top-k step. External kernels here (RAFT top-k,
  FlashInfer's `top_k_page_table_transform`) ARE drop-in candidates
  because they read the existing score buffer and write logical or
  physical indices into a known output. Evaluate them via the
  `_BaseSelector` contract.
- **Score phase**: K-label × Q-label dot product. External kernels
  here are NOT drop-in candidates as long as our K-label cache is flat
  and theirs is paged. Either we paginate (large coordinated change)
  or we rewrite the kernel against our layout (defeats the purpose of
  borrowing).

### Anti-patterns

- "FlashInfer kernel X has a fused FOO+BAR — let's wire it." Without
  first checking that X's cache layout matches DS's flat row-indexed
  K-label, the integration is doomed to a per-step transform that eats
  the win.
- Measuring only the kernel's micro-µs without measuring the
  surrounding plumbing cost (layout xform, allocator pressure, capture
  break). The `microbench_ftka_backends.py` script records
  `layout_transform_us` explicitly to prevent this.
- "We'll just allocate page_size=1 paged buffers each step." Each
  allocation breaks CUDA graph capture and triggers `cudaMallocAsync`
  fragmentation; the DS hot path is allergic to both.

### Verification signals

After wiring an external kernel into a microbench, the integration is
**measurably safe** if:

- `parity_match = ok` against the torch baseline on small shapes.
- `graph_status = ok` under an isolated `torch.cuda.graph` probe.
- `(kernel_us + layout_transform_us) < baseline_us` at the production
  shape (bs=32, ctx=128K for the 70B/TP=8 / DS workload).

The integration is **structurally unsafe** if:

- The kernel uses a cache layout you don't already have.
- The transform from your layout to theirs requires per-step
  reallocation or per-step copies that scale with `bs × max_ctx`.
- The kernel needs JIT-compiled Triton internals that may bind to a
  specific stream context (see
  [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]).

## When to Use

Read this when:

- Considering ANY external sparse-attention kernel (FlashInfer, FTKA,
  vLLM custom CUDA, sgl-kernel new ops) as a substitute for DS's
  score or selector phase.
- Reviewing a benchmark that claims "X is faster than torch.topk" or
  "X is faster than our score kernel" — verify the benchmark includes
  the layout transform cost.
- Designing a future paged K-label cache (for a different reason, e.g.
  prefix-cache integration). At that point, this analysis flips:
  paged external kernels become drop-in candidates and the *flat*
  ones become structurally adverse.

## Context Links
- Based on: [[knowledge/ds-flashinfer-top-k-page-table-boundaries/content]]
- Based on: [[knowledge/ds-native-sparse-decode-pareto/content]]
- Leads to: [[decisions/2026-05-17-ftka-evaluation-scoped-to-selector-only]]
- Related: [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]
