# Dual-Chunk Attention Capability Matrix

This folder covers dual-chunk attention tests. `dual_chunk_flash_attn` is not
a dense backend swap: it expects a packed five-way query projection (`query`,
`succ`, `inter`, and critical variants), so the dense Q/K/V harness is
structurally wrong for this method. The single attention backend here is
`dual_chunk_flash_attn`; the rows below distinguish kernel-path modes.

## Coverage Matrix

Columns are runner modes; rows are kernel-path modes of the single
`dual_chunk_flash_attn` backend. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Kernel path | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Non-sparse | ✓ first-window, successor-chunk, inter-chunk extend/decode layouts + GQA decode | deferred: graph metadata for dual-chunk not scoped | deferred | deferred | blocked: `init_forward_metadata` asserts `is_prefill() or is_decode()` (`dual_chunk_flashattention_backend.py:179`); `TARGET_VERIFY` falls under `is_prefill()` but the wrapper hasn't been wired through | deferred | deferred | deferred | blocked: `DRAFT_EXTEND_V2` excluded from `is_prefill()` alias (see Production-Unsupported below) | deferred | deferred | — |
| Sparse all-column (`vertical_size`/`slash_size` chosen so every key in the first chunk is selected) | ✓ single-request first-chunk, multi-request first-chunk, page-boundary first-chunk | — | — | — | blocked: same `is_prefill` assertion | — | — | — | blocked: same | — | — | — |
| Threshold-gated sparse (`sparse_attention_threshold=100`, seq_len=16 → gate disables sparse, falls back to dense) | ✓ verifies `current_orig_seq_len > threshold` gate semantics | — | — | — | — | — | — | — | — | — | — | — |

## Input And Config Coverage

- Page size 1 extend, exact-page extend, page-boundary crossing extend, and
  ragged extend batches.
- Decode page-boundary coverage and GQA decode coverage.
- Successor-chunk and inter-chunk extend/decode layouts where `query_succ`
  and `query_inter` are active and use independent projection weights.
- Sparse all-column prefill uses `head_dim=128` to match the local sparse
  FlashAttention build and selects every column in the first chunk
  (≤16 tokens) so the dense reference remains valid.
- Multi-request sparse and page-boundary sparse variants exercise per-request
  `cu_seqlens_*` slicing inside `_dual_chunk_flash_attn_prefill_func`.
- Threshold-gated sparse uses `sparse_attention_threshold=100` so a 16-token
  prompt bypasses the sparse kernel and falls through to the dense chunk
  flash path, exercising the gate semantics in the wrapper.

## Production-Unsupported

- **Non-prefill / non-decode forward modes** —
  `dual_chunk_flashattention_backend.py:179` asserts
  `forward_mode.is_prefill() or forward_mode.is_decode()`. `is_prefill()`
  aliases to `is_extend()` (`forward_batch_info.py:103-104`) and covers
  `EXTEND` / `MIXED` / `DRAFT_EXTEND` / `TARGET_VERIFY` / `SPLIT_PREFILL` /
  `DLLM_EXTEND`, but `DRAFT_EXTEND_V2` is excluded by default. So
  `DRAFT_EXTEND_V2` is structurally unreachable for `dual_chunk_flash_attn`.
- **Non-causal / windowed-attention requests** — `forward_extend` raises
  `ValueError("Dual Chunk Attention does not support causal=False")`
  (`dual_chunk_flashattention_backend.py:698`) and
  `ValueError("Dual Chunk Attention does not support window_size")`
  (`dual_chunk_flashattention_backend.py:700`).
- **Sparse mode `chunk_len % block_size != 0`** — raises
  `ValueError("chunk_len must be divisible by block_size.")`
  (`dual_chunk_flashattention_backend.py:860, 1491`). The current fixture
  picks divisible values.
- **Unsupported `head_dim`** — only `head_dim in {16, 32, 64, 128, 256, 512}`
  is accepted (`dual_chunk_flashattention_backend.py:1611`).

## Next Work

- Populate CUDA graph and PCG/BCG runner metadata after eager non-sparse
  coverage is stable across more chunk layouts.
- Extend sparse-attention and critical-token reference coverage beyond the
  all-column sparse kernel path (i.e., real sub-context-window pruning that
  diverges from the dense reference).
