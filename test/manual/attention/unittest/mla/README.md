# MLA Attention Capability Matrix

This folder covers absorb-style DeepSeek MLA attention. The actual path writes
latent KV through `get_token_to_kv_pool()` before calling `attn_mqa`; expected
outputs come from a separate HF-style PyTorch MLA reference with copied random
weights and no SGLang backend calls.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | Full representative MLA input-shape sweep | CUDA graph decode; PCG/BCG extend | EAGLE chain verify; EAGLE tree CUDA graph verify; production EAGLE draft decode chain/tree; production EAGLE `DRAFT_EXTEND_V2` graph replay | Regular `DRAFT_EXTEND` is not enabled for Triton MLA. |
| `flashinfer` | Full representative MLA sweep with DeepSeek-like `kv_lora_rank=512`, `qk_rope_head_dim=64` | CUDA graph decode; PCG/BCG extend | EAGLE chain verify and CUDA graph verify; production EAGLE draft decode; production EAGLE `DRAFT_EXTEND` | EAGLE tree verify currently mismatches the PyTorch reference. |
| `flashmla` | FlashMLA-compatible page-size-64 MLA cases with DeepSeek-like shape metadata | CUDA graph decode; PCG/BCG extend | EAGLE chain verify and CUDA graph verify; production EAGLE draft decode; EAGLE `DRAFT_EXTEND` eager | `DRAFT_EXTEND` CUDA graph capture is blocked by missing `cuda_graph_qo_indptr` in the inherited FlashInfer MLA path. |
| `cutlass_mla` | Not enabled | Not enabled | Not enabled | Local SM90 reports compute capability 10.0 requirement. |
| `trtllm_mla` | Not enabled | Not enabled | Not enabled | Local path reports SM120a/SM121a requirement. |
| `tokenspeed_mla` | Not enabled | Not enabled | Not enabled | Needs an FP8 KV-cache fixture (`kv_cache_dtype=fp8_e4m3`). |

## Input And Config Coverage

- Page size 1, page-boundary decode, exact-page and crossing-page extend cases.
- Ragged page-boundary extend batches.
- Representative page-size-32 crossing case.
- Nonzero MLA rope dimension support is present, but RoPE math is intentionally
  orthogonal to the runner/backend matrix.

## Next Work

- Debug FlashInfer MLA tree custom-mask semantics.
- Fix or work around FlashMLA draft-extend graph metadata allocation.
- Add hardware-gated tests for `cutlass_mla`, `trtllm_mla`, and `tokenspeed_mla`
  when appropriate hardware/KV dtype fixtures are available.
