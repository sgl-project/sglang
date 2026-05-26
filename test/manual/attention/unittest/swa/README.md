# Sliding Window Attention Capability Matrix

This folder covers dense attention with a finite `sliding_window_size`.
Expected outputs use the dense HF-style PyTorch reference with sliding-window
masking, not a second backend call.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | No-prefix lengths below/equal/above window; prefix lengths below/equal/above window | CUDA graph decode within the configured window; PCG/BCG extend | EAGLE `TARGET_VERIFY` chain/tree; EAGLE tree CUDA graph replay | Current graph tests avoid above-window decode because that exposes a semantic mismatch before graph replay. |
| `flashinfer` | No-prefix lengths below/equal/above window | CUDA graph decode within the configured window; PCG/BCG extend | Not enabled | FlashInfer target-verify metadata currently expects prefix data that the synthetic target-verify path does not supply. |
| `torch_native` | No-prefix and prefix window edges; decode and GQA decode window edges | Eager only | Not enabled | Torch-native SDPA now uses an explicit local-attention mask for finite windows; no CUDA graph or speculative coverage. |

## Current Progress

- SWA has representative Phase 2 and Phase 3 coverage for `triton` and `flashinfer`.
- Phase 4 coverage is intentionally concentrated on `triton`, where the synthetic
  verify metadata path currently matches the independent reference.
- `torch_native` is covered for eager Phase 2 SWA cases only because it has no CUDA
  graph implementation.

## Next Work

- Build a more faithful FlashInfer SWA target-verify fixture before enabling speculative tests.
- Investigate the Triton above-window decode/reference mismatch separately from runner coverage.
