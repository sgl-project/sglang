# Dual-Chunk Attention Capability Matrix

This folder covers dual-chunk attention tests. `dual_chunk_flash_attn` is not a
dense backend swap: it expects a packed five-way query projection (`query`,
`succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dual_chunk_flash_attn` non-sparse | Eager prefill/decode coverage for first-window, successor-chunk, inter-chunk, and GQA decode layouts | Not implemented | Not implemented | Uses a method-specific packed-query fixture and an independent PyTorch reference that merges intra, successor, and inter groups by global softmax. |
| `dual_chunk_flash_attn` sparse all-column | Three eager prefill layouts: single-request, multi-request first-chunk, and page-boundary first-chunk | Not implemented | Not implemented | Sparse `vertical_size`/`slash_size` chosen so the kernel selects every key in the first chunk, keeping the dense PyTorch reference valid. |
| `dual_chunk_flash_attn` threshold-gated sparse | One eager prefill layout where `current_orig_seq_len <= sparse_attention_threshold` | Not implemented | Not implemented | Verifies that `sparse_attention_enabled=True` with `sparse_attention_threshold=100` correctly falls back to dense prefill when seq_len is below threshold. |

## Input And Config Coverage

- Page size 1 extend, exact-page extend, page-boundary crossing extend, and
  ragged extend batches.
- Decode page-boundary coverage and GQA decode coverage.
- Successor-chunk and inter-chunk extend/decode layouts where `query_succ`
  and `query_inter` are active and use independent projection weights.
- Sparse all-column prefill coverage uses `head_dim=128` to match the local
  sparse FlashAttention build and selects every column in the first chunk
  (≤16 tokens) so the dense reference remains valid.
- Multi-request sparse and page-boundary sparse variants exercise per-request
  `cu_seqlens_*` slicing inside `_dual_chunk_flash_attn_prefill_func` under
  sparse enabled.
- Threshold-gated sparse uses `sparse_attention_threshold=100` so a 16-token
  prompt bypasses the sparse kernel and falls through to the dense chunk
  flash path, exercising the gate semantics in the wrapper.

## Current Progress

- Phase 2 eager coverage is enabled for first-window, successor-chunk, and
  inter-chunk non-sparse dual-chunk layouts.
- Phase 2 eager coverage is enabled for three sparse all-column prefill
  layouts (single-request, multi-request first-chunk, page-boundary
  first-chunk) plus a threshold-gated layout.
- The fixture uses the real `dual_chunk_flash_attn` backend with method-
  specific packed projections and an independent PyTorch reference that
  merges intra, successor, and inter groups by global softmax semantics.
- Runner and speculative coverage remain intentionally deferred until sparse
  and graph metadata behavior are scoped.

## Next Work

- Populate CUDA graph and PCG/BCG runner metadata after eager non-sparse
  coverage is stable across more chunk layouts.
- Extend sparse-attention and critical-token reference coverage beyond the
  all-column sparse kernel path (i.e., real sub-context-window pruning that
  diverges from the dense reference).
