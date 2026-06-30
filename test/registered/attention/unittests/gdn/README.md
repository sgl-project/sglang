# GDN Attention Capability Matrix

The main matrix uses the backend in each row for **full attention** and keeps
the GDN linear-attention kernel on Triton. A separate focused class covers
FlashInfer GDN prefill. Its basic output and final-state case uses the
pure-PyTorch reference; its tracked-state checkpoint case uses Triton as oracle.

## Coverage Matrix

Columns are runner modes; rows are full-attention backends (linear-attention
kernel = `triton` for all rows). Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable (no production path for this combination)
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Full-attn backend | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch_native` | ✓ full representative GDN input sweep | — (no CG hooks on `TorchNativeAttnBackend`) | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | — | — | — | — | — | — | — | — |
| `triton` | ✓ full representative GDN input sweep | ✓ decode page-boundary | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) + EAGLE tree (topk=2) | ✓ EAGLE chain + EAGLE tree (tree uses scoped `5e-2` atol for bf16 recurrent accumulation) | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | — | blocked: same `_replay_metadata` reject | — |
| `flashinfer` | ✓ full GDN sweep with `head_dim=64` (FlashInfer SM90 prefill constraint) | ✓ decode page-boundary | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) + EAGLE tree (topk=2) | ✓ EAGLE chain + EAGLE tree (scoped `5e-2` atol) | — | blocked: same `_replay_metadata` reject | blocked: same `_replay_metadata` reject | — | blocked: same `_replay_metadata` reject | — |

### FlashInfer linear-GDN coverage

`TestFlashInferLinearGDNBackendCorrectness` explicitly selects FlashInfer only
for GDN prefill and checks a ragged nonzero-prefix case, final recurrent-state
writeback, tracked-state checkpoints, and chain/tree target verification.

## Hybrid dispatch fan-out tests (Triton only, MagicMock-based)

These cover the `HybridLinearAttnBackend` dispatch layer itself (not numerical
correctness). Each test constructs a `HybridLinearAttnBackend` with two
`MagicMock` sub-backends and asserts both receive the matching call.

| Test | Mutation covered |
|---|---|
| `test_hybrid_dispatch_eager_init_forward_metadata_fan_out` | M20 — `attn_backend_list[1:]` slice in `init_forward_metadata` (`hybrid_linear_attn_backend.py:825-827`) |
| `test_hybrid_dispatch_replay_init_forward_metadata_fan_out` | M19 — `attn_backend_list[:1]` slice in `init_forward_metadata_replay_cuda_graph` (`hybrid_linear_attn_backend.py:879-900`) |
| `test_hybrid_dispatch_capture_init_forward_metadata_fan_out` | Symmetric capture coverage (not in mutation journal) |

## Input And Config Coverage

- Page size 1, exact-page, crossing-page, ragged page-boundary, page-size-32
  crossing, decode boundary, and batch-size-1 decode cases.
- GDN uses speculative Mamba state buffers for target verify coverage.
- The split-op tests verify live-token slicing with a larger static token
  buffer.

## Production-Unsupported

- **HybridLinearAttnBackend CUDA-graph capture/replay outside
  `DECODE_OR_IDLE` / `TARGET_VERIFY`** — `MambaAttnBackendBase._capture_metadata`
  / `_replay_metadata` (`hybrid_linear_attn_backend.py:493-572`) raise
  `ValueError(f"Invalid forward mode: {forward_mode=}")` for anything else.
  This is the underlying contract for GDN's `Mamba2AttnBackend`, KDA,
  Lightning, and Mamba2. So `DRAFT_EXTEND` / `DRAFT_EXTEND_V2` CUDA-graph
  capture/replay is structurally unreachable for the GDN linear-attention
  side.
- **HybridLinearAttnBackend `_forward_metadata` modes** — same file
  (`hybrid_linear_attn_backend.py:246`): non-decode, non-extend modes raise
  `ValueError`. Legal modes are `is_decode_or_idle`, plus
  `is_extend(include_draft_extend_v2=True)` (which subsumes `EXTEND` /
  `MIXED` / `DRAFT_EXTEND` / `DRAFT_EXTEND_V2` / `TARGET_VERIFY` /
  `SPLIT_PREFILL` / `DLLM_EXTEND` per `forward_batch_info.py:106-115`).

## Caveats

- **Prefix continuation uses a synthetic recurrent state.**
  `build_gdn_attention_fixture` calls `_populate_gdn_prefix_state` to seed a
  deterministic, nonzero per-request SSM state whenever `prefix_lens > 0`.
  Actual and reference paths therefore exercise consumption and update of a
  nontrivial initial state. The fixture does not derive that state by running
  the prefix tokens through the module, so it does not independently validate
  prefix-state construction.

## Next Work

- Add additional linear-attention kernel backend variants when available.
- Consider broader speculative worker tags only after EAGLE chain/tree remains
  stable across kernels.
