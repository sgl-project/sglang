# KDA Attention Capability Matrix

This folder covers KDA (Kimi Delta Attention) linear attention. The actual
path drives `KDAAttnBackend` through `HybridLinearAttnBackend` on a
`RadixLinearAttention` layer. Expected outputs come from an independent pure
PyTorch sigmoid-gated delta-rule reference using `KimiLinearCacheParams` /
`KimiLinearStateShape` (per-head-channel `dt_bias`, `silu` activation on
conv1d output, per-channel gate broadcast), not the KDA Triton kernel.

## Current Matrix

| Linear-attention kernel backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | Full representative KDA input-shape sweep | Not implemented | Not implemented | Reference recomputes the gated-delta recurrence in pure PyTorch with the same random conv1d / norm / projection weights as the actual path. |

## Input And Config Coverage

- Page size 1, exact-page, crossing-page, ragged page-boundary,
  page-size-32 crossing, decode page-boundary, and batch-size-1 decode
  cases (10 input variants from `make_kda_cases`).
- `num_k_heads=2, num_v_heads=2` with head-dims defaulted by
  `DEFAULT_HEAD_K_DIM = DEFAULT_HEAD_V_DIM = 32`.

## Current Progress

- Phase 2 eager correctness covers the representative KDA input layouts
  against a pure PyTorch sigmoid-gated delta-rule recurrence reference.
- Runner and speculative coverage are intentionally deferred. KDA's
  HybridLinearAttnBackend dispatch and recurrent-state-aware capture/replay
  will land alongside the GDN-style coverage once a representative kernel
  contract is selected.

## Production-Unsupported

- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN, so
  the `ValueError("Invalid forward mode")` at
  `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:509,572`
  rejects `DRAFT_EXTEND` / `DRAFT_EXTEND_V2` / `EXTEND` graph runners. Any
  Phase 4 KDA draft-extend graph runner is structurally unreachable.

## Next Work

- Add CUDA graph decode and PCG/BCG runner coverage modeled on the existing
  GDN runner-mode helpers.
- Add EAGLE chain/tree speculative target-verify coverage once the recurrent
  state buffer setup is mirrored from the GDN fixture.
- Consider additional KDA kernel backend variants when available.
