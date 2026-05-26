# Dual-Chunk Attention Capability Matrix

This folder covers dual-chunk attention tests. `dual_chunk_flash_attn` is not a
dense backend swap: it expects a packed five-way query projection (`query`,
`succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dual_chunk_flash_attn` | Eager prefill/decode coverage for first-window, successor-chunk, inter-chunk, GQA decode, and sparse all-column layouts | Not implemented | Not implemented | Uses a method-specific packed-query fixture and independent PyTorch references for non-sparse grouping plus a compact sparse-kernel path. |

## Input And Config Coverage

- Page size 1 extend, exact-page extend, page-boundary crossing extend, and
  ragged extend batches.
- Decode page-boundary coverage and GQA decode coverage.
- Successor-chunk and inter-chunk extend/decode layouts where `query_succ` and
  `query_inter` are active and use independent projection weights.
- Sparse all-column prefill coverage uses `head_dim=128` to match the local sparse
  FlashAttention build and selects every column in a 16-token prompt so the dense
  reference remains valid.
- Critical query variants are present in the packed tensor; sparse critical-token
  pruning beyond the all-column case is still outside the current reference.

## Current Progress

- Phase 2 eager coverage is enabled for first-window, successor-chunk, and
  inter-chunk non-sparse dual-chunk layouts, plus one sparse all-column prefill
  layout.
- The fixture uses the real `dual_chunk_flash_attn` backend with method-specific
  packed projections and an independent PyTorch reference that merges intra,
  successor, and inter groups by global softmax semantics.
- Runner and speculative coverage remain intentionally deferred until sparse and
  graph metadata behavior are scoped.

## Next Work

- Populate CUDA graph and PCG/BCG runner metadata after eager non-sparse coverage
  is stable across more chunk layouts.
- Extend sparse-attention and critical-token reference coverage beyond the
  all-column sparse kernel path.
