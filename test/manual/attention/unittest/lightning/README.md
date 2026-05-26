# Lightning Attention Capability Matrix

This folder covers Bailing-style segmented linear attention (`seg_la`). The
actual path wraps `RadixAttention` and installs `LightningAttentionBackend`
directly via `ForwardContext`, since Lightning's layer wrapper is plain
`RadixAttention` and `HybridLinearAttnBackend` would route it to the full
backend. Expected outputs come from an independent pure PyTorch per-token
`seg_la` recurrence reference (`state_t = state_{t-1} * exp(-slope_h) +
outer(k_t, v_t)`, `o_t = q_t @ state_t * head_dim**-0.5`).

## Current Matrix

| Linear-attention kernel backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | Full representative Lightning input-shape sweep | Not implemented | Not implemented | Reference uses an independent ALiBi-slope decay recurrence with the same random projection weights. |

## Input And Config Coverage

- Page size 1, exact-page, crossing-page, ragged page-boundary,
  page-size-32 crossing, decode page-boundary, and batch-size-1 decode
  cases (10 input variants from `make_lightning_cases`).
- `num_heads=2` with `DEFAULT_HEAD_DIM=128`. Head dim is intentionally 128
  because the `seg_la` Triton kernels constrain it:
  - decode (`seg_la_d_kernel`): `K_SPLIT_DIM=128`, so `head_dim >= 128`.
  - prefill with `bs > 2` (`seg_la_p_kernel`): `V_SPLIT_DIM=64`, so
    `head_dim >= 64`.

## Current Progress

- Phase 2 eager correctness covers the representative Lightning input
  layouts against the pure PyTorch reference.
- Runner and speculative coverage are intentionally deferred.

## Production-Unsupported

- **`raise ValueError` paths in `LightningAttentionBackend`** —
  `python/sglang/srt/layers/attention/linear/lightning_backend.py:332` and
  `lightning_backend.py:369` reject configurations the seg_la kernels do not
  support; the head-dim assertions above are the practical entry-point
  guards.
- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  Lightning inherits the `MambaAttnBackendBase` capture/replay contract, so
  the same `ValueError("Invalid forward mode")` raises at
  `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:509,572`
  apply. Draft-extend graph runners are structurally unreachable.

## Next Work

- Add CUDA graph decode and PCG/BCG runner coverage. Lightning's recurrent
  state lives in the same `Mamba2CacheParams.temporal` layout as GDN, so the
  capture/replay flow should be adaptable from `cuda_graph_decode_runner`.
- Add EAGLE chain/tree speculative target-verify coverage with the matching
  recurrent-state buffer reset between capture and replay.
