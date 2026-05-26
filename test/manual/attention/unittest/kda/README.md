# KDA Attention Capability Matrix

This folder covers KDA (Kimi Delta Attention) linear attention. The actual
path drives `KDAAttnBackend` through `HybridLinearAttnBackend` on a
`RadixLinearAttention` layer. Expected outputs come from an independent
pure-PyTorch sigmoid-gated delta-rule reference using
`KimiLinearCacheParams` / `KimiLinearStateShape` (per-head-channel `dt_bias`,
`silu` activation on conv1d output, per-channel gate broadcast), not the KDA
Triton kernel.

## Coverage Matrix

Columns are runner modes; rows are the linear-attention kernel backend
(`triton` is the only one wired today). Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Linear-attn kernel | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ decode page-boundary (uses `KDA_GRAPH_ATOL=1e-1` to absorb Triton recurrent-kernel CG-replay drift; eager `KDA_ATOL=3e-2` kept for non-graph cases) | deferred | deferred | deferred: `expected_kda_verify_output_from_inputs` reads `inputs["a"]/["b"]` as raw `[T, HV*K]` / `[T, HV]` shapes (matching `fixture.a_raw`/`b_raw`), but `kda_fixture_inputs` and `make_kda_random_inputs` return the *shaped* (`[1, T, HV, K]`) / per-head-scalar (`[T, HV]`) versions used by `run_kda_forward`. Verify wiring needs `make_kda_random_inputs` + `kda_fixture_inputs` to expose raw a/b alongside the shaped versions, then `expected_kda_verify_output_from_inputs` adjusted to consume the raw key. | deferred | — | blocked: HybridLinearAttnBackend `_replay_metadata` rejects modes outside `DECODE_OR_IDLE` / `TARGET_VERIFY` (`hybrid_linear_attn_backend.py:509,572`) | blocked: same `_replay_metadata` reject | deferred | blocked: same `_replay_metadata` reject | — |

## Input And Config Coverage

- 10 input variants from `make_kda_cases('triton')`: page 1, exact-page,
  crossing-page, ragged page-boundary, page-size-32 crossing, decode
  page-boundary, batch-size-1 decode.
- `num_k_heads=2, num_v_heads=2` with head dims defaulted by
  `DEFAULT_HEAD_K_DIM = DEFAULT_HEAD_V_DIM = 32`.

## Production-Unsupported

- **CUDA-graph capture/replay outside `DECODE_OR_IDLE` / `TARGET_VERIFY`** —
  KDA inherits the same `MambaAttnBackendBase` capture/replay path as GDN,
  so `ValueError("Invalid forward mode")` at
  `hybrid_linear_attn_backend.py:509, 572` rejects `DRAFT_EXTEND` /
  `DRAFT_EXTEND_V2` / `EXTEND` graph runners. Any Phase 4 KDA draft-extend
  graph runner is structurally unreachable.

## Next Work

- Add PCG/BCG runner coverage modeled on GDN `runner_modes`. CG decode
  already wired via `run_kda_cuda_graph_decode_case` (see above).
- Add EAGLE chain/tree speculative target-verify coverage. Blocked by
  the `expected_kda_verify_output_from_inputs` ↔ `make_kda_random_inputs`
  shape mismatch documented in the matrix above; the existing verify
  reference path is correct for `fixture.a_raw / b_raw`, but the inputs
  dict surfaced by `kda_fixture_inputs` carries the *shaped* tensors
  that `run_kda_forward` passes to the actual module. Fix needs
  `make_kda_random_inputs` to expose both shapes so capture inputs can
  feed both the forward and the reference, then the verify
  expected-output wrapper consumes the raw key.
- Consider additional KDA kernel backend variants when available.
