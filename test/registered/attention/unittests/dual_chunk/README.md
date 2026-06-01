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

## Container Gate (SM10.x)

`DualChunkFlashAttentionBackend` calls `flash_attn_varlen_func` via
`sglang.jit_kernel.flash_attention`. On SM8.x / SM9.x that resolves to sgl-kernel's
FA3 build; on SM != {8, 9} (notably SM10.x / GB300) the JIT kernel falls back
to the upstream `flash_attn` (FA2) wheel, which the
`lmsysorg/sglang:nightly-dev-cu13` container ships without an SM10.x-compiled
`flash_attn_varlen_func`. `test_dual_chunk_flash_attn.py` probes both paths at
module import: if FA3 is supported (`major in {8, 9}`) it runs unconditionally;
otherwise it tries `from flash_attn import flash_attn_varlen_func` and skips the
whole class with the documented reason if the symbol is missing. Re-image with
an SM10.x-compiled flash_attn wheel to clear; no test-code change needed.

See `KNOWN_FAILURES.md` §1 for the full root cause + fix.

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
- **Sub-context-window sparse pruning reference (genuine follow-up)** —
  The current "all-column" sparse cases match the dense reference exactly
  because the chosen `vertical_size=16` + `slash_size=16` + `last_q=16`
  configuration covers every column in the first chunk for `seq_len <= 16`.
  A truly pruning case needs `seq_len >> vertical_size + slash_size` and a
  reference that applies the same mask the kernel applies.

  The blocker is that the production sparse-attention config
  `("vertical_and_slash", v_size, s_size, threshold)` is **content-aware**:
  per-head `v_idx` and `s_idx` are picked by top-k attention scores over
  the last `last_q` queries, not from a fixed schedule
  (`dual_chunk_flashattention_backend.py:_dual_chunk_flash_attn_prefill`).
  An independent reference therefore has three paths:

  1. **Mock the sparse-config lookup** — patch
     `get_sparse_attention_config` or the per-layer top-k selection so the
     fixture supplies known `v_idx` / `s_idx` tensors. Then write a
     token-level reference that masks `attn_scores[q, k] = -inf` unless
     `k in v_idx` or `(q - k) in s_idx` (with causal `k <= q`). This is the
     cleanest path but needs a hook in `_dual_chunk_flash_attn_prefill_func`
     that doesn't exist today.
  2. **Replicate `convert_vertical_slash_indexes`** at block granularity in
     pure-PyTorch, then iterate `(block_count, block_offset, column_count,
     column_index)` to build a per-(query_block, key_block) mask matching
     the kernel's selection. Faithful but tedious — the block math (M=64,
     N=64) needs to be mirrored exactly.
  3. **Statistical recovery check** — compute dense attention scores
     `softmax(Q @ K^T)` per head, identify the top-k columns by score, and
     verify the sparse kernel output approximates the dense output modulo
     the dropped probability mass. Not strict `assert_close`; rejects only
     gross divergences.

  Option 1 is recommended. It requires either: (a) a new
  `sparse_attention_config_override` kwarg threaded through
  `DualChunkFlashAttentionBackend.__init__` that bypasses the content-aware
  selection, or (b) monkeypatching `get_sparse_attention_config` on the
  fixture's backend instance. Until that lands, the all-column sparse +
  threshold-gated cases keep the kernel/wrapper integration covered but
  the per-column sparse math is unverified.

  **Production-side bugs surfaced while attempting Option 3
  (smoke-test "sparse output != dense output"):** two issues block even a
  smoke-only sub-window test today.

  - `dual_chunk_flashattention_backend.py:1110-1122`: when a chunk's
    `intra_vertical_indices.nelement() == 0`, the fallback appends
    `torch.arange(0, intra_K_size, max(1, intra_K_size/5))`. With
    `intra_K_size=48` this is `arange(0, 48, 9.6)` → 5 elements, but the
    `vertical_buffer` is sized to `vertical_size` (=4 in a sub-window
    config). The copy at line 1132 then raises
    `RuntimeError: The size of tensor a (4) must match the size of
    tensor b (5)`. The fallback should clip to `vertical_size` slots.
  - With `vertical_size=8` to clear the overflow, the sparse kernel
    crashes with `cudaErrorIllegalAddress` deep inside
    `_vertical_slash_sparse_attention`, suggesting the
    `convert_vertical_slash_indexes` block math has an unstated
    invariant that `vertical_size + slash_size >= chunk_len_blocks` or
    similar. Needs a kernel-side audit.

  The smoke-test helper `run_dual_chunk_sparse_sub_window_case` is wired
  through `common/attention_methods/dual_chunk_attention.py` for when
  those production bugs are fixed; no test method invokes it today.
