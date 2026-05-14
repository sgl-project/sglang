---
id: ds-flashinfer-top-k-page-table-boundaries-content
type: knowledge
title: flashinfer.top_k_page_table_transform — boundaries, performance, capture-incompatibility
status: active
created: 2026-05-14
updated: 2026-05-14
tags: [flashinfer, top-k, cuda-graph, sparse-decode, double-sparsity]
---

# `flashinfer.top_k_page_table_transform` — boundaries

Empirical findings from wiring this kernel into the DS native sparse-decode
selector backend (flashinfer 0.6.11.post1, sm90 H200, CUDA 13.2).

## Signature (for reference)

```python
flashinfer.top_k_page_table_transform(
    input: Tensor,                # [num_rows, max_len]  fp32 / fp16 / bf16 scores
    src_page_table: Tensor,       # [batch_size, max_len]  int32  (e.g. req_to_token)
    lengths: Tensor,              # [num_rows]  int32  per-row history bound
    k: int,                       # top-k count
    row_to_batch: Optional[Tensor] = None,   # [num_rows]  int32; if None, identity
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
    row_starts: Optional[Tensor] = None,
) -> Tensor  # [num_rows, k] int32 of physical page positions
```

Output semantics:
> `output[i, j] = src_page_table[batch_idx_for_row_i, topk_indices[j]]`
> where `batch_idx_for_row_i = row_to_batch[i]` if provided else `i`.

So this kernel **fuses top-k + page-table lookup**. For per-h_kv broadcast,
flatten scores to `[bs*h_kv, max_ctx]` and supply `row_to_batch = i // h_kv`.

## Boundaries

### B1. Hard `top_k <= 2048` ceiling

Requests with `top_k > 2048` raise CUDA `operation not supported` from
inside the kernel. The bound is independent of `max_ctx` (verified at
`max_ctx in {16384, 131072}`). The kernel does NOT cleanly degrade
to a slower path above the ceiling — it just errors.

**Implication for callers**: validate at config time. The DS runtime
config rejects `selector_backend='flashinfer_topk_page_table'` paired
with `token_budget > FLASHINFER_TOPK_MAX` (2048).

### B2. CUDA-graph-capture incompatibility (current installed env)

The kernel **crashes inside SGLang's CUDA graph capture region** with
`Triton Error [CUDA]: illegal memory access` in `load_binary` even
after a pre-capture warmup sweep that JIT-compiles the kernel at every
captured bs in the SGLang ladder `{1, 2, 4, 8, 12, 16, 24, 32}`.

Diagnosis: warmup at all bs completes successfully (logged
"DoubleSparsity selector warmup OK"), then the FIRST captured forward
call crashes in `load_binary`. The symptom suggests Triton's compiled
kernel handle is bound to the stream/context it was first compiled on;
the capture-stream context triggers a re-load that fails.

`dsa_graph_safe=True` does NOT fix this. The flag's documented behavior
is "force FilteredTopK path and graph-safe vectorization (VEC_SIZE=1)",
which addresses kernel-internal vectorization concerns, not the
inter-context load_binary issue.

**Workaround**: don't use this backend under graph replay. The DS
selector backend registry exposes it for microbench / future use, but
the production decode path defaults to `selector_backend='torch'`.

### B3. Performance vs `torch.topk` (when usable)

70B/TP=8 shape (h_kv=1 per rank, max_ctx=131072), single-stream
microbench excluding capture overhead:

|   bs |  top_k | torch.topk (µs) | flashinfer (µs) | speedup |
|-----:|-------:|----------------:|----------------:|--------:|
|    1 |  1024  |     88          |    84           |  1.05×  |
|    4 |  1024  |     87          |    85           |  1.02×  |
|    8 |  2048  |     87          |    86           |  1.01×  |
|   16 |  1024  |     94          |    87           |  1.09×  |
|   16 |  2048  |     96          |    87           |  1.10×  |
|   32 |  1024  |    114          |    87           |  1.31×  |
|   32 |  2048  |    115          |    89           |  1.30×  |

Speedup grows with batch: small batches are launch-overhead-bound on
both backends; at bs >= 16 the FlashInfer fused selector pulls ahead.
At bs=32 / tb=2048 the per-step saving is ~2.1 ms across 80 layers —
meaningful relative to a typical bs=32 TBT around 30 ms.

`torch.topk` decomposes into 8 distinct `at::native::mbtopk::*` +
`scan_by_key::DeviceScan*` kernels per call (totalling ~0.64 s of
stream-time across the conc=32 / tb=8192 nsys trace). FlashInfer
collapses this to one launch.

## Conditions for invalidation

* `flashinfer.top_k_page_table_transform` may raise the top-k ceiling
  in a future release. Re-measure at top_k in {4096, 8192} when
  upgrading.
* Triton + CUDA-graph capture compatibility may improve. Re-test by
  running `benchmark/double_sparsity/bench_decode.py
  --selector-backend flashinfer_topk_page_table` end-to-end (not just
  the microbench, which doesn't exercise capture).

## Context Links
- Based on: [[knowledge/ds-native-sparse-decode-pareto/content]]
- Related: [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]
