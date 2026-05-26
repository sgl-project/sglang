# MLA Attention Capability Matrix

This folder covers absorb-style DeepSeek MLA attention. The actual path writes
latent KV through `get_token_to_kv_pool()` before calling `attn_mqa`; expected
outputs come from a separate HF-style PyTorch MLA reference with copied random
weights and no SGLang backend calls.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `triton` | Full representative MLA input-shape sweep | CUDA graph decode; PCG/BCG extend | EAGLE chain verify; EAGLE tree CUDA graph verify; production EAGLE draft decode chain/tree; production EAGLE `DRAFT_EXTEND_V2` graph replay | Regular `DRAFT_EXTEND` is not enabled for Triton MLA. |
| `flashinfer` | Full representative MLA sweep with DeepSeek-like `kv_lora_rank=512`, `qk_rope_head_dim=64` | CUDA graph decode; PCG/BCG extend | EAGLE chain verify and CUDA graph verify; production EAGLE draft decode (chain only); production EAGLE `DRAFT_EXTEND` | EAGLE tree verify (`topk=2`) is production-unsupported — see below. |
| `flashmla` | FlashMLA-compatible page-size-64 MLA cases with DeepSeek-like shape metadata | CUDA graph decode; PCG/BCG extend | EAGLE chain verify and CUDA graph verify; production EAGLE draft decode (chain only); EAGLE `DRAFT_EXTEND` eager | `DRAFT_EXTEND` CUDA graph capture is blocked by missing `cuda_graph_qo_indptr` in the inherited FlashInfer MLA path. EAGLE tree verify (`topk=2`) is production-unsupported — see below. |
| `cutlass_mla` | Not enabled | Not enabled | Not enabled | Local SM90 reports compute capability 10.0 requirement. Cutlass MLA also has no `forward_extend` and only supports `is_decode_or_idle` (`cutlass_mla_backend.py:86,156,197`); tree verify (`topk=2`) inherits the FlashInfer-MLA reject when the draft side wires it. |
| `trtllm_mla` | Not enabled | Not enabled | Not enabled | Local path reports SM120a/SM121a requirement. Multi-step draft backend inherits `topk == 1` reject — see below. |
| `tokenspeed_mla` | Not enabled | Not enabled | Not enabled | Needs an FP8 KV-cache fixture (`kv_cache_dtype=fp8_e4m3`, enforced at `server_args.py:2814-2818`) and SM100+. Multi-step draft backend inherits `topk == 1` reject — see below. |

## Input And Config Coverage

- Page size 1, page-boundary decode, exact-page and crossing-page extend cases.
- Ragged page-boundary extend batches.
- Representative page-size-32 crossing case.
- Nonzero MLA rope dimension support is present, but RoPE math is intentionally
  orthogonal to the runner/backend matrix.

## Current Progress

- Phase 2 eager correctness covers the locally runnable MLA backends with a
  separate absorb-style PyTorch reference.
- Phase 3 coverage includes CUDA graph decode and paged/chunked extend replay
  for Triton, FlashInfer MLA, and FlashMLA.
- Phase 4 now includes both synthetic EAGLE verify coverage and production
  EAGLE draft-runner graph replay for the stable backend/mode combinations.

## Production-Unsupported

These combinations are explicitly rejected by the production speculative
multi-step draft backends and cannot ever appear at runtime. They are recorded
here so future test work does not add them as "next-step" follow-ups.

- **FlashInfer MLA tree verify / draft-extend with `topk > 1`** — raised by
  `FlashInferMLAMultiStepDraftBackend.__init__` at
  `python/sglang/srt/layers/attention/flashinfer_mla_backend.py:910-913`:
  `if topk > 1: raise ValueError("Currently Flashinfer MLA only supports topk=1
  for speculative decoding")`. The draft side of any FlashInfer MLA spec run is
  hard-gated, so MLA EAGLE tree (`topk=2`) on FlashInfer cannot be reached in
  production. The dispatcher routes MLA EAGLE through this class at
  `python/sglang/srt/speculative/draft_utils.py:126-132`.
- **FlashMLA tree verify / draft-extend with `topk > 1`** — raised by
  `FlashMLAMultiStepDraftBackend.__init__` at
  `python/sglang/srt/layers/attention/flashmla_backend.py:555-558`:
  `if topk > 1: raise ValueError("Currently FlashMLA only supports topk=1 for
  speculative decoding")`. Dispatched from
  `python/sglang/srt/speculative/draft_utils.py:173-180`.
- **TRT-LLM MLA tree verify / draft-extend with `topk > 1`** —
  `TRTLLMMLAMultiStepDraftBackend` inherits from
  `FlashInferMLAMultiStepDraftBackend`
  (`python/sglang/srt/layers/attention/trtllm_mla_backend.py:1223-1229`) so the
  same `topk > 1` `raise ValueError` fires before any TRT-LLM-specific code
  runs. Dispatched from `python/sglang/srt/speculative/draft_utils.py:191-203`.
- **Tokenspeed MLA tree verify / draft-extend with `topk > 1`** —
  `TokenspeedMLAMultiStepDraftBackend` inherits from
  `TRTLLMMLAMultiStepDraftBackend`
  (`python/sglang/srt/layers/attention/tokenspeed_mla_backend.py:341-347`), so
  the same chain-only ValueError applies.
- **Tokenspeed MLA non-FP8 KV cache** — `tokenspeed_mla` requires
  `kv_cache_dtype=fp8_e4m3` (server-side check at
  `python/sglang/srt/server_args.py:2814-2818`) and constructor rejection at
  `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py:111-113`
  (`if self.page_size not in (32, 64): raise ValueError`). The MLA fixture's
  mock model runner does not currently emit FP8 KV cache.
- **Cutlass MLA extend / verify / draft-extend modes** — `CutlassMLABackend`
  only overrides `forward_decode`
  (`python/sglang/srt/layers/attention/cutlass_mla_backend.py:226`) and only
  handles `is_decode_or_idle` in `init_forward_metadata*`
  (`cutlass_mla_backend.py:86,156,197`). Any non-decode mode falls through to
  `super()` (FlashInfer MLA), so the only Cutlass-specific paths are decode.
  Hardware-gated on SM 10.0 (Blackwell).
- **All MLA backends fixed page size** — FlashMLA forces `page_size=64`
  (`server_args.py:2767-2770`); Cutlass MLA forces `page_size=128`
  (`server_args.py:2776-2779`, also hard-coded as `PAGE_SIZE=128` at
  `cutlass_mla_backend.py:31`); TRT-LLM MLA and Tokenspeed MLA force
  `page_size in {32, 64}` (`server_args.py:2790-2794` and `2809-2813`).

## Next Work

- Fix or work around FlashMLA draft-extend graph metadata allocation.
- Add hardware-gated tests for `cutlass_mla`, `trtllm_mla`, and `tokenspeed_mla`
  decode (chain spec only) when appropriate hardware/KV dtype fixtures are
  available.
