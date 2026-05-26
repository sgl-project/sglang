# Mamba2 / SSM Attention Capability Matrix

This folder covers Mamba2 state-space-model attention. The actual path
constructs a real `MambaMixer2` and drives it through `Mamba2AttnBackend` via
`ForwardContext`. Expected outputs come from a pure-PyTorch per-token SSM scan
reference (`state_t = exp(A*dt_t) * state_{t-1} + dt_t * B_t * x_t`,
`y_t = C_t * state_t + D * x_t`) that reuses the actual `in_proj` / `conv1d` /
`norm` / `out_proj` modules through shared random weights but recomputes the
SSM core entirely in pure torch.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` (`Mamba2AttnBackend`) | Eager EXTEND zero-prefix exact-page (16 tokens) | Not implemented | Not implemented | Tolerance loosened to `5e-2` to absorb chunked-scan reordering and bf16 `out_proj` accumulation depth. |

## Input And Config Coverage

- Page size 16 zero-prefix EXTEND with 16 tokens.
- `num_heads=DEFAULT_NUM_HEADS=2`, `head_dim=DEFAULT_HEAD_DIM=16`,
  `state_size=16`, `n_groups=1`, `conv_kernel=4`,
  `mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE=16`, `hidden_size=32`.
- Dims chosen as the minimum that satisfies `MambaMixer2`'s TP/chunk asserts.

## Current Progress

- Phase 2 eager correctness is enabled for the single representative EXTEND
  case.
- Runner and speculative coverage are not implemented yet; broader input-
  shape coverage (page boundary, ragged, decode) requires the
  `MambaAttnBackendBase` dispatch path and Mamba2 cache parameter setup to
  be wired through the fixture.

## Production-Unsupported

- **`Mamba2AttnBackend.forward_decode` / `forward_extend` raise** —
  `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:743-749`
  raises `NotImplementedError` for direct `forward_decode` / `forward_extend`
  on `Mamba2AttnBackend`. Production dispatches through
  `HybridLinearAttnBackend.forward_extend` /
  `HybridLinearAttnBackend.forward_decode`
  (`hybrid_linear_attn_backend.py:899-917,868-886`) instead.
- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  the underlying `MambaAttnBackendBase` capture/replay rejects all other
  modes (`hybrid_linear_attn_backend.py:509,572`).
- **Per-mixer head_dim / chunk constraints** — `MambaMixer2.__init__` asserts
  weight dimension sums (`mamba.py:92`), TP head divisibility
  (`mamba.py:217,221,226`), and ssd kernels reject mismatched group / chunk
  shapes (`ops/ssd_chunk_state.py:448-509,576-583`). These are config-time
  asserts and the test fixture explicitly sets dims to satisfy them.

## Required Fixture Work

- Wire the `HybridLinearAttnBackend` dispatch wrapper into the fixture so
  the production `init_forward_metadata*` paths and per-layer dispatch are
  actually exercised (currently the fixture installs `Mamba2AttnBackend`
  directly via `ForwardContext`).
- Add a CUDA graph decode fixture with explicit recurrent cache snapshot/
  restore between capture and replay, matching the GDN runner-mode shape.

## Next Work

- Expand to additional input shapes (page boundary, ragged extend, decode).
- Add `HybridLinearAttnBackend` dispatch coverage.
- Add CUDA graph decode replay with recurrent state isolation.
