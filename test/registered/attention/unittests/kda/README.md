# KDA Attention Capability Matrix

This folder covers KDA (Kimi Delta Attention) linear attention. The actual
path drives `KDAAttnBackend` through `HybridLinearAttnBackend` on a
`RadixLinearAttention` layer. Expected outputs come from an independent
pure-PyTorch sigmoid-gated delta-rule reference using
`KimiLinearCacheParams` / `KimiLinearStateShape` (per-head-channel `dt_bias`,
`silu` activation on conv1d output, per-channel gate broadcast), not the KDA
Triton kernel.

## Coverage Matrix

Columns are runner modes; rows are the linear-attention kernel backend. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Linear-attn kernel | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ decode page-boundary (uses `KDA_GRAPH_ATOL=1e-1` to absorb Triton recurrent-kernel CG-replay drift; eager `KDA_ATOL=3e-2` kept for non-graph cases) | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) + EAGLE tree (topk=2) (`atol=1e-1` because the verify reference's pure-Python per-token recurrence drifts ~0.07 vs the Triton kernel even before CG capture/replay) | ✓ EAGLE chain CG + EAGLE tree CG (same `1e-1` tolerance) | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |
| `flashinfer` prefill (SM100/SM103, FlashInfer 0.7.0) | ✓ KDA backend extend with beta logits, 128-D heads, and aligned/unaligned radix-cache snapshots; kernel tests cover BF16/FP32 and indexed state pools | — | — | — | — | — | — | — | — | — | — | — |

## Input And Config Coverage

- 10 input variants from `make_kda_cases('triton')`: page 1, exact-page,
  crossing-page, ragged page-boundary, page-size-32 crossing, decode
  page-boundary, batch-size-1 decode.
- `num_k_heads=2, num_v_heads=2` with head dims defaulted by
  `DEFAULT_HEAD_K_DIM = DEFAULT_HEAD_V_DIM = 32`.
- `test_kda_prefill_flashinfer.py` exercises the 128-D FlashInfer path against
  Triton on B200/GB300, including the 130-token to 128-token cache boundary.

## Production-Unsupported

- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN,
  so `ValueError("Invalid forward mode")` at
  `hybrid_linear_attn_backend.py:509, 572` rejects `DRAFT_EXTEND` /
  `DRAFT_EXTEND_V2` / `EXTEND` graph runners. Any Phase 4 KDA draft-extend
  graph runner is structurally unreachable.

## Next Work

- Add model-level Kimi K3 coverage when weights are available in the runner.
