# Lightning Attention Capability Matrix

This folder covers Bailing-style segmented linear attention (`seg_la`). The
actual path wraps `RadixAttention` and installs `LightningAttentionBackend`
directly via `ForwardContext`, since Lightning's layer wrapper is plain
`RadixAttention` and `HybridLinearAttnBackend` would route it to the full
backend. Expected outputs come from an independent pure-PyTorch per-token
`seg_la` recurrence reference (`state_t = state_{t-1} * exp(-slope_h) +
outer(k_t, v_t)`, `o_t = q_t @ state_t * head_dim**-0.5`).

## Coverage Matrix

Columns are runner modes; rows are the linear-attention kernel backend
(`triton` is the only one wired today). Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Linear-attn kernel | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | deferred: Hybrid CG decode + recurrent-cache snapshot path not wired | deferred | deferred | deferred: needs Hybrid recurrent-state buffer wiring | deferred | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |

## Input And Config Coverage

- 10 input variants from `make_lightning_cases('triton')`: page 1,
  exact-page, crossing-page, ragged page-boundary, page-size-32 crossing,
  decode page-boundary, batch-size-1 decode.
- `num_heads=2` with `DEFAULT_HEAD_DIM=128`. Head dim is intentionally 128
  because the `seg_la` Triton kernels constrain it:
  - decode (`seg_la_d_kernel`): `K_SPLIT_DIM=128`, so `head_dim >= 128`.
  - prefill with `bs > 2` (`seg_la_p_kernel`): `V_SPLIT_DIM=64`, so
    `head_dim >= 64`.

## Production-Unsupported

- **`raise ValueError` paths in `LightningAttentionBackend`** —
  `lightning_backend.py:332, 369` reject configurations the seg_la kernels
  do not support; the head-dim constraints above are the practical
  entry-point guards.
- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  Lightning inherits the `MambaAttnBackendBase` capture/replay contract, so
  `ValueError("Invalid forward mode")` at `hybrid_linear_attn_backend.py:509,
  572` applies. Draft-extend graph runners are structurally unreachable.

## Next Work

- Add CUDA graph decode + PCG/BCG runner coverage. Lightning's recurrent
  state lives in the same `Mamba2CacheParams.temporal` layout as GDN, so the
  capture/replay flow should be adaptable from `cuda_graph_decode_runner`.
- Add EAGLE chain/tree speculative target-verify coverage with the matching
  recurrent-state buffer reset between capture and replay.
