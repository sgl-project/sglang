# Sliding Window Attention Capability Matrix

This folder covers dense attention with a finite `sliding_window_size`.
Expected outputs use the dense HF-style PyTorch reference with sliding-window
masking, not a second backend call.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | No-prefix lengths below/equal/above window; prefix lengths below/equal/above window | CUDA graph decode within the configured window; PCG/BCG extend | EAGLE `TARGET_VERIFY` chain/tree; EAGLE tree CUDA graph replay | Current graph tests avoid above-window decode because that exposes a semantic mismatch before graph replay. |
| `flashinfer` | No-prefix lengths below/equal/above window | CUDA graph decode within the configured window; PCG/BCG extend | EAGLE `TARGET_VERIFY` is production-unsupported in the current implementation — see below. |  |
| `torch_native` | No-prefix and prefix window edges; decode and GQA decode window edges | Eager only | Not enabled | Torch-native SDPA now uses an explicit local-attention mask for finite windows; no CUDA graph or speculative coverage. |

## Current Progress

- SWA has representative Phase 2 and Phase 3 coverage for `triton` and `flashinfer`.
- Phase 4 coverage is intentionally concentrated on `triton`, where the synthetic
  verify metadata path currently matches the independent reference.
- `torch_native` is covered for eager Phase 2 SWA cases only because it has no CUDA
  graph implementation.

## Production-Unsupported

- **FlashInfer SWA `TARGET_VERIFY` with the current SWA prefill updater** —
  `FlashInferIndicesUpdaterPrefill.update_sliding_window`
  (`python/sglang/srt/layers/attention/flashinfer_backend.py:1296-1344`)
  computes per-wrapper paged lens as `min(seq_lens, window + seq_lens -
  prefix_lens)`. If `prefix_lens is None` (which is what
  `init_forward_metadata` / `init_forward_metadata_replay_cuda_graph` pass for
  target-verify and draft-extend at `flashinfer_backend.py:742,754`), the
  subtraction raises. So FlashInfer SWA target-verify and draft-extend cannot
  reach the SWA prefill kernel without a separate fix to the prefill metadata
  contract. Move from "Next Work" to here.
- **`torch_native` SWA speculative / CUDA graph** — torch-native backend has
  no CUDA-graph capture/replay path (see `dense/README.md`), so all
  Phase 3 and Phase 4 graph integration is structurally unsupported.

## Next Work

- Investigate the Triton above-window decode/reference mismatch separately from runner coverage.
- The FlashInfer SWA verify path would need a new metadata contract that
  threads `prefix_lens` through the target-verify replay; until that lands the
  fixture is intentionally inactive.
