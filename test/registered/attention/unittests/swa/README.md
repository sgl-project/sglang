# Sliding Window Attention Capability Matrix

This folder covers dense attention with a finite `sliding_window_size`.
Expected outputs use the dense HF-style PyTorch reference with sliding-window
masking, not a second backend call. The SWA fixture is the dense fixture
reused with `sliding_window_size != None`.

## Coverage Matrix

Columns are runner modes; rows are attention backends. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable (no production path for this combination)
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Backend | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch_native` | ✓ no-prefix + prefix window edges, MHA + GQA decode window edges (uses explicit SDPA local-attention mask) | — (no CG hooks) | — (no CG path) | — (no CG path) | — | — | — | — | — | — | — | — |
| `triton` | ✓ no-prefix lengths below/equal/above window + prefix lengths below/equal/above window | ✓ within-window decode (`prefix_lens=(1,2,3)`, `window=4`) + above-window decode (`prefix_lens=(7,8,9)`, `window=4`) | ✓ no-prefix window edges, prefix-within-window MHA extend | ✓ same as PCG | ✓ EAGLE chain (topk=1) + EAGLE tree (topk=2), `window=4` | ✓ EAGLE tree within-window + EAGLE chain above-window (`prefix_lens=(6,8)`, `window=4`) | — | — | — | — | — | — |
| `flashinfer` | ✓ no-prefix lengths below/equal/above window (`head_dim=64` for SM90) | ✓ within-window decode | ✓ no-prefix window edges (MHA extend) | ✓ same as PCG | ✓ DFLASH chain (`topk=1`, `window=4`) | ✓ DFLASH chain (`topk=1`, `window=4`) | — | — | — | — | — | — |

## Input And Config Coverage

- No-prefix lengths below / equal / above the configured `sliding_window_size`.
- For `triton`: matching prefix-length cases.
- For `torch_native`: extra MHA + GQA decode cases at the window edge.
- CG decode covers both within-window (`min(seq_lens, window)` clipped) and
  above-window (full window clip) for `triton`.

## Notes on the "—" cells

- **`torch_native` graph rows** — same as dense: no CUDA-graph capture/replay
  hooks (`base_attn_backend.py:24-55` raises `NotImplementedError`).
- **SWA-only methods** — DSV4 SWA, DSA dense fallback, and other SWA-shaped
  paths live in their own folders. This folder is strictly the dense MHA/GQA
  backend with a finite window.

## Mutation Coverage Notes

- The CG-decode above-window case (`runner_cuda_graph_swa_decode_above_window`)
  exists specifically to expose the `sliding_window_size + 1` mutation at
  `triton_backend.py:786` (M5). The dense reference picks the matching SWA mask
  rule based on `case.backend in _SWA_AWARE_DECODE_BACKENDS` and
  `case.forward_mode.is_decode()`.
- The Verify CG above-window case
  (`runner_cuda_graph_eagle_verify_swa_above_window`) extends above-window
  coverage to the verify replay path, but does not catch M6 by itself — the
  extend kernel re-masks `kv_id >= q_id - sliding_window_size` so the +1 shift
  the mutation introduces is dropped. See `MUTATION_FIXES.md`.

## Production-Unsupported

- **FlashInfer SWA `DRAFT_EXTEND`** — not covered here. The FlashInfer SWA
  coverage added for this path is limited to DFLASH `TARGET_VERIFY`.
- **`torch_native` SWA speculative / CUDA graph** — no CG hooks; all graph
  integration is structurally unsupported.

## Next Work

- Investigate the Triton above-window decode/reference numerical detail
  separately (the above-window case currently asserts within tolerance with the
  matching reference rule; if a real backend regression appears, lower the
  tolerance).
