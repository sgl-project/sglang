# Dual-Chunk Attention Capability Matrix

This folder is reserved for dual-chunk attention tests. `dual_chunk_flash_attn`
is not a dense backend swap: it expects a packed five-way query projection
(`query`, `succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method.

## Planned Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dual_chunk_flash_attn` | Short-sequence eager prefill/decode coverage inside the first dual-chunk window | Not implemented | Not implemented | Uses a method-specific packed-query fixture and an independent dense causal PyTorch reference while succ/inter chunks are inactive. |

## Required Fixture Work

- Extend coverage beyond first-chunk semantics so succ/inter chunk query streams
  and dual-chunk output merging are checked against a dedicated PyTorch reference.
- Populate CUDA graph and PCG/BCG runner metadata after eager first-chunk coverage
  is stable across more chunk layouts.
- Add sparse-attention reference coverage once a compact sparse config is selected.

## First Test Target

- Broaden `dual_chunk/test_dual_chunk_flash_attn.py` from first-chunk equivalence
  to cross-chunk layouts that require `query_succ` and `query_inter`.
- Add Phase 3/4 only after the full packed-query reference is stable.
