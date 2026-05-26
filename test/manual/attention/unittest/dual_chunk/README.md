# Dual-Chunk Attention Capability Matrix

This folder covers dual-chunk attention tests. `dual_chunk_flash_attn` is not a
dense backend swap: it expects a packed five-way query projection (`query`,
`succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dual_chunk_flash_attn` | Short-sequence eager prefill/decode coverage inside the first dual-chunk window | Not implemented | Not implemented | Uses a method-specific packed-query fixture and an independent dense causal PyTorch reference while succ/inter chunks are inactive. |

## Input And Config Coverage

- Page size 1 extend, exact-page extend, page-boundary crossing extend, and
  ragged extend batches.
- Decode page-boundary coverage and GQA decode coverage.
- First-window chunk layouts where `query_succ`, `query_inter`, and critical
  variants are present in the packed tensor but inactive in the reference.

## Current Progress

- Phase 2 eager coverage is enabled for the first dual-chunk window.
- The fixture uses the real `dual_chunk_flash_attn` backend with method-specific
  packed projections and an independent dense causal PyTorch reference.
- Runner and speculative coverage are intentionally blocked until cross-chunk
  semantics have a dedicated reference.

## Next Work

- Extend coverage beyond first-chunk semantics so succ/inter chunk query streams
  and dual-chunk output merging are checked against a dedicated PyTorch reference.
- Populate CUDA graph and PCG/BCG runner metadata after eager first-chunk coverage
  is stable across more chunk layouts.
- Add sparse-attention reference coverage once a compact sparse config is selected.
