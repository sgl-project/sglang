# Dense Attention Capability Matrix

This folder covers standard dense MHA/GQA/MQA attention through `RadixAttention`.
Expected outputs come from independent HF-style PyTorch reference modules with
copied random projection weights, not from another SGLang attention backend.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `torch_native` | Full dense input-shape sweep plus MHA/GQA/MQA configs | Eager decode/extend runner checks | Not enabled | `TARGET_VERIFY` probe currently fails in torch-native SDPA extend metadata handling. |
| `triton` | Full dense input-shape sweep plus MHA/GQA/MQA configs | CUDA graph decode for MHA/GQA/MQA; PCG/BCG extend for MHA/GQA | EAGLE/Frozen-KV-MTP/DFlash/NGRAM verify; verify CUDA graph for EAGLE/DFlash/NGRAM; EAGLE `DRAFT_EXTEND_V2` CUDA graph | `DRAFT_EXTEND` is intentionally blocked by a reference mismatch. |
| `flashinfer` | Full dense sweep with `head_dim=64` for SM90 prefill constraints | CUDA graph decode for MHA/GQA/MQA; PCG/BCG extend for MHA/GQA | EAGLE/Frozen-KV-MTP/DFlash/NGRAM verify; verify CUDA graph for EAGLE/Frozen-KV-MTP/DFlash; EAGLE/Frozen-KV-MTP `DRAFT_EXTEND` eager and CUDA graph | `DRAFT_EXTEND_V2` is not enabled yet. |
| `fa3` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Decode/speculative CUDA graph replay currently mismatches the PyTorch reference. |
| `fa4` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Same current graph blocker as `fa3`. |
| `flex_attention` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Backend mask callback compatibility is covered; graph/spec coverage is still open. |
| `trtllm_mha` | Decode-only MHA/GQA/MQA and page-size-32 coverage | Not enabled | Not enabled | Local SM90 decode passes; prefill reports unsupported architecture and graph replay mismatches. |

## Input And Config Coverage

- Page size 1, page size 16, and representative page size 32.
- Zero-prefix exact page, prefix exact page, total exact page, and page-boundary crossing.
- Ragged batches with lengths below/equal/above a page.
- Decode page-boundary batches and batch-size-1 decode.
- Attention config coverage for MHA, GQA, and MQA is separate from input-layout coverage.

## Next Work

- Debug Triton `DRAFT_EXTEND` metadata/reference mismatch.
- Debug FA3/FA4 CUDA graph replay and speculative graph mismatches.
- Add backend-specific graph coverage for `trtllm_mha` once local hardware and metadata behavior allow it.
