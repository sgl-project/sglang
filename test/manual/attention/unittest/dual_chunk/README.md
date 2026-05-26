# Dual-Chunk Attention Capability Matrix

This folder is reserved for dual-chunk attention tests. `dual_chunk_flash_attn`
is not a dense backend swap: it expects a packed five-way query projection
(`query`, `succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method.

## Planned Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dual_chunk_flash_attn` | Not implemented | Not implemented | Not implemented | Needs a method-specific fixture and independent PyTorch reference. |

## Required Fixture Work

- Build a tiny model-specific module that emits the packed dual-chunk query streams.
- Implement an HF-style reference for the dual-chunk mask/layout semantics.
- Populate real request/KV pools and `ForwardBatch` metadata instead of adapting the dense fixture.

## First Test Target

- `dual_chunk/test_dual_chunk_flash_attn.py` with Phase 2 eager correctness over page-boundary inputs.
- Add Phase 3/4 only after the packed-query reference and eager path are stable.
