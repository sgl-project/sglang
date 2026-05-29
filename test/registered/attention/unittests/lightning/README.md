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
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ decode page-boundary (uses `LIGHTNING_GRAPH_ATOL=1e-1` to absorb seg_la kernel CG-replay drift; eager `LIGHTNING_ATOL=3e-2` kept for non-graph cases) | deferred: piecewise CG path returns per-head shape via `RadixAttention.forward`'s `empty_like(q)`, but Lightning backend's `forward_extend` flattens to `[T, num_heads * head_dim]`; eager vs piecewise actuals don't share a shape. See "Production-Unsupported" below. | deferred (same reason) | ✓ EAGLE chain (topk=1) only — see "Production-Unsupported" below for why tree is omitted. Uses `atol=1e-1` because the verify reference's pure-Python per-token recurrence drifts ~0.07 vs the seg_la Triton kernel. | ✓ EAGLE chain CG (same `1e-1` tolerance) | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |

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
- **EAGLE tree (topk>1) verify** — `seg_la.py` has no parent-indices /
  retrieve-index plumbing; the kernel processes draft tokens as a chain
  regardless of the input tree shape. A tree-shaped verify produces
  large divergence (~5x off) vs the parent-indices-aware reference. The
  `intermediate_state_indices` / `intermediate_ssm` plumbing in
  `lightning_backend.py:307-329` is per-request, not per-token, so it
  cannot replay parent state forks. Only chain (topk=1) is covered.
- **PCG / BCG split-op extend** — Lightning's `forward_extend` flattens
  to `[T, num_heads * head_dim]` at `lightning_backend.py:335`, but
  under piecewise CG `RadixAttention.forward`
  (`radix_attention.py:124-137`) writes through `output =
  torch.empty_like(q)` of per-head shape `[T, num_heads, head_dim]`,
  ignoring the backend's intended flatten. The shared
  `_run_split_op_extend_case` compares eager vs piecewise actuals,
  which then trip a shape mismatch. KDA and GDN avoid this because
  their backends keep the per-head shape on the return path. Fixing
  needs either a Lightning-specific split-op runner that reshapes
  actual to flat, or a Lightning backend change to keep per-head shape
  under piecewise CG.

## Next Work

- PCG/BCG split-op extend needs either a Lightning-specific split-op
  runner that reshapes piecewise actual to flat, or a backend-side
  change to keep per-head shape under piecewise CG. See
  "Production-Unsupported" above.
- EAGLE tree verify is gated by the `seg_la` kernel itself (no
  parent-indices support); landing it requires a kernel-side change to
  thread parent indices through `intermediate_ssm` so each draft token
  forks from its parent's saved state rather than the prior chain
  position. Out of scope for unit tests.
