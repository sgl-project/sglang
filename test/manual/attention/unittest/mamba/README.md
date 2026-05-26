# Mamba2 / SSM Attention Capability Matrix

This folder covers Mamba2 state-space-model attention. The actual path
constructs a real `MambaMixer2` and drives it through `Mamba2AttnBackend` via
`ForwardContext`. Expected outputs come from a pure-PyTorch per-token SSM scan
reference (`state_t = exp(A*dt_t) * state_{t-1} + dt_t * B_t * x_t`,
`y_t = C_t * state_t + D * x_t`) that reuses the actual `in_proj` / `conv1d` /
`norm` / `out_proj` modules through shared random weights but recomputes the
SSM core entirely in pure torch.

## Coverage Matrix

Columns are runner modes; rows are the SSM kernel backend
(`triton` `Mamba2AttnBackend` is the only one wired today). Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **metadata-only** — backend exercised through the metadata path only (no
  forward), used to cover specific mutation surfaces
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| SSM kernel | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` (`Mamba2AttnBackend`) | ✓ EXTEND zero-prefix exact-page (16 tokens) | metadata-only: `init_forward_metadata_replay_cuda_graph` with `seq_lens_cpu=[5,1,1]` to cover the M21 padding-count mutation in `_replay_metadata`; full forward replay blocked by baseline `enable_symm_mem` bug | deferred | deferred | deferred | deferred | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |

## Input And Config Coverage

- Page size 16 zero-prefix EXTEND with 16 tokens.
- `num_heads=DEFAULT_NUM_HEADS=2`, `head_dim=DEFAULT_HEAD_DIM=16`,
  `state_size=16`, `n_groups=1`, `conv_kernel=4`,
  `mamba_chunk_size=DEFAULT_MAMBA_CHUNK_SIZE=16`, `hidden_size=32`.
- Dims chosen as the minimum that satisfies `MambaMixer2`'s TP/chunk asserts.
- Replay metadata test uses `prefix_lens=(4, 0, 0)` and feeds
  `seq_lens_cpu=[5, 1, 1]` directly so two trailing rows match the
  CUDA-graph fill value (`1`).

## Production-Unsupported

- **`Mamba2AttnBackend.forward_decode` / `forward_extend` raise** —
  `hybrid_linear_attn_backend.py:743-749` raises `NotImplementedError` for
  direct calls. Production dispatches through `HybridLinearAttnBackend`'s
  forward (`hybrid_linear_attn_backend.py:899-917, 868-886`).
- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  the underlying `MambaAttnBackendBase` capture/replay rejects all other
  modes (`hybrid_linear_attn_backend.py:509, 572`).
- **Per-mixer head_dim / chunk constraints** — `MambaMixer2.__init__` asserts
  weight dim sums (`mamba.py:92`), TP head divisibility (`mamba.py:217, 221,
  226`), and ssd kernels reject mismatched group / chunk shapes
  (`ops/ssd_chunk_state.py:448-509, 576-583`). The fixture sets dims to
  satisfy these.

## Known Baseline Issue

- `test_projected_mamba2_attention_cases` currently fails on the foreground
  repo with `'SimpleNamespace' object has no attribute 'enable_symm_mem'`.
  Root cause: the mock `server_args` `SimpleNamespace` lacks
  `enable_symm_mem`, which the production `is_symmetric_memory_enabled()`
  reads via `get_global_server_args()` inside `MambaMixer2.in_proj` /
  `out_proj`. The M21 mutation test (`test_mamba2_replay_metadata_padding_indices`)
  intentionally goes metadata-only to side-step this.

## Required Fixture Work

- Wire the `HybridLinearAttnBackend` dispatch wrapper into the fixture so
  production `init_forward_metadata*` paths and per-layer dispatch are
  actually exercised (today the fixture installs `Mamba2AttnBackend`
  directly via `ForwardContext`).
- Add a CUDA graph decode fixture with explicit recurrent cache snapshot /
  restore between capture and replay, matching the GDN runner-mode shape.

## Next Work

- Expand to additional input shapes (page boundary, ragged extend, decode).
- Add `HybridLinearAttnBackend` dispatch coverage.
- Add CUDA graph decode replay with recurrent state isolation.
