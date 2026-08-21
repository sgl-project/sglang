# MLA Attention Capability Matrix

This folder covers absorb-style DeepSeek MLA attention. The actual path writes
latent KV through `get_token_to_kv_pool()` before calling `attn_mqa`; expected
outputs come from a separate HF-style PyTorch MLA reference with copied random
weights and no SGLang backend calls.

## Coverage Matrix

Columns are runner modes; rows are attention backends. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable (no production path for this combination)
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled
- **skip:hw** — hardware-gated; skipped on this environment but enabled when
  the gating predicate passes

| Backend | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `triton` | ✓ 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ MLA decode page-boundary | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) | ✓ EAGLE tree (topk=2) | — (V1 DE not enabled for Triton MLA; Triton uses V2 path) | — | ✓ fixed-tokens-per-req | ✓ chain (topk=1) + tree (topk=2) | ✓ via `DRAFT_EXTEND_V2` graph runner | — (no FKVMTP wiring for MLA) |
| `flashinfer` | ✓ 10 input layouts with DeepSeek-like `kv_lora_rank=512`, `qk_rope_head_dim=64` | ✓ MLA decode page-boundary | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) | ✓ EAGLE chain (topk=1) | ✓ EAGLE ragged-accept | ✓ EAGLE ragged-accept | blocked: `is_draft_extend()` default `include_v2=False` (`flashinfer_mla_backend.py:432,501,454-455,512`) | ✓ chain (topk=1) only — tree blocked by `topk=1` reject (`flashinfer_mla_backend.py:910-913`) | ✓ EAGLE ragged-accept (V1) | — (no FKVMTP wiring for MLA) |
| `flashmla` | ✓ FlashMLA-compatible page-size-64 cases (zero-prefix exact page, input page edges 63/64/65, prefix exact page, total exact page, cross page, ragged, decode page-boundary, decode bsz=1 nonzero prefix) | ✓ page-size-64 decode page-boundary | ✓ ragged page-boundary extend | ✓ ragged page-boundary extend | ✓ EAGLE chain (topk=1) | ✓ EAGLE chain (topk=1) | ✓ EAGLE ragged-accept | deferred: parent FlashInfer-MLA capture path expects 1D `cuda_graph_kv_indices`, FlashMLA allocates 2D `[max_bs, (max_context+PAGE_SIZE)//PAGE_SIZE]` (`flashmla_backend.py:347-348` + parent `init_forward_metadata_capture_cuda_graph`) | — (FlashMLA does not implement V2) | ✓ chain (topk=1) only — tree blocked by `topk=1` reject (`flashmla_backend.py:555-558`) | — (DE CG deferred above) | — |
| `cutlass_mla` | skip:hw — needs SM 10.0+ (Blackwell); current 1 case uses `ForwardMode.EXTEND` but `CutlassMLABackend` only overrides `forward_decode` (`cutlass_mla_backend.py:226`) and falls through to FlashInfer MLA for other modes → **case should be DECODE**; PAGE_SIZE fixed at 128 (`cutlass_mla_backend.py:31`) | — (decode-only backend; no extend/CG) | — | — | blocked: tree via `topk=1` reject inherited from FlashInfer MLA parent | — | — | — | — | — | — | — |
| `trtllm_mla` | skip:hw — needs SM 12.0a / 12.1a (`is_sm120_supported`) | — | ✓ 9 varlen-absorbed cases, FP8 KV, page 32/64 (zero-prefix, exact page, cross page, ragged bs=2/3/4 incl. a 1-token extend); skip:hw — SM 10.x only, see below | ✓ same 9 varlen-absorbed cases under `enable_breakable_cuda_graph()`; skip:hw — SM 10.x only, see below | blocked: `topk=1` only (`trtllm_mla_backend.py:1223-1229` inherits from FlashInfer MLA) | — | — | — | — | — | — | — |
| `tokenspeed_mla` | skip:hw — needs `find_spec("tokenspeed_mla")`, SM 10.0+, and `kv_cache_dtype=fp8_e4m3` (`server_args.py:2814-2818`); current MLA fixture does not emit FP8 KV cache | — | — | — | blocked: `topk=1` only (`tokenspeed_mla_backend.py:341-347` inherits from TRT-LLM MLA) | — | — | — | — | — | — | — |

## Input And Config Coverage

- Page size 1, page-boundary decode, exact-page and crossing-page extend cases.
- Ragged page-boundary extend batches.
- Representative page-size-32 crossing case (`triton`, `flashinfer`).
- FlashMLA cases use `page_size=64` because `FlashMLABackend` forces that size
  (`server_args.py:2767-2770`). The 8 FlashMLA EXTEND/DECODE input
  variants cover zero-prefix exact-page, input page edges
  (`extend=(63, 64, 65)`), prefix exact-page (`prefix=64`), total
  exact-page (`prefix=32, extend=32`), cross-page-boundary
  (`prefix=63, extend=2`), ragged page-boundary
  (`prefix=(0, 32, 64), extend=(63, 32, 1)`), decode page-boundary,
  and decode bsz=1 nonzero-prefix.
- Nonzero MLA rope dimension support is present in the fixture, but RoPE math
  is intentionally orthogonal to the runner/backend matrix.

## `trtllm_mla` PCG/BCG Extend (Varlen Absorbed MLA)

Under a captured `tc_piecewise` **or** `breakable` prefill graph, the
DeepSeek/Kimi dispatcher returns `AttnForwardMethod.MLA` even for a genuine
extend, so `trtllm_mla` serves it with absorbed MLA over a ragged `q`
(`_run_varlen_absorbed_kernel`) instead of the FlashInfer paged-MLA parent.
`test_trtllm_mla_piecewise.py` covers the `tc_piecewise` half,
`test_trtllm_mla_breakable.py` covers the `breakable` half (same case matrix
and tolerances, entering `enable_breakable_cuda_graph()` instead) —
`TRTLLMMLABackend.use_varlen_absorbed` is
`(is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph()) and ...`,
so both halves of that OR need independent coverage. `supports_varlen_absorbed_mla`
is the per-backend opt-out and is contract-tested in
`test_mla_varlen_absorbed_gate.py`; it applies uniformly to both capture modes.

Post-#34245 ("Allow prefill breakable CUDA graph for the Kimi archs"), Kimi's
prefill defaults to `breakable`, not `tc_piecewise` — so the `breakable` half
is what the default path actually exercises now, not just defensive coverage.

- **Batch multiplicity** is covered at bs=1/2/3/4. The sizes are not arbitrary:
  instrumenting a Kimi-K2.6 GSM8K run showed this branch serving bs=1 30256x,
  bs=2 4392x, bs=3 488x and bs=4 244x, while the prefill graph had only ever
  captured bs=1. The bs=4 cases include a 1-token extend beside a 64-token one,
  the widest spread `cum_seq_lens_q` has to describe.
- **Tolerance** is `atol = rtol = 2e-1`, looser than `MLA_ATOL`, to absorb the
  native FP8 trtllm-gen kernel's error against the BF16 reference. The margin is
  thin: the measured max abs error on these fixtures is ~1.8e-1. A block-table
  shift of one page moves it to ~1.3e3, so the tolerance still separates a real
  defect by ~4 orders of magnitude.
- **Not covered:** capture/replay of a real prefill CUDA graph, for either
  capture mode. The `test_trtllm_mla_piecewise.py` fixture only sets
  `prefill=PhaseConfig(backend=TC_PIECEWISE)`, and `test_trtllm_mla_breakable.py`
  only enters `enable_breakable_cuda_graph()` — both take the production branch
  but run eager and assert the eager output. Batches padded to a captured
  bucket, and the breakable graph's own segmented-capture/replay mechanics
  (distinct from `tc_piecewise`'s torch.compile-based capture), are therefore
  out of scope here and are only exercised end-to-end.
- **Environment gate:** `varlen_absorbed_mla_supported()` — SM 10.x, non-FP4 KV.
  flashinfer resolves `backend="auto"` to XQA when the compute-capability major
  is not 10, and XQA rejects `cum_seq_lens_q`; FP4 KV is excluded because
  `MLATokenToKVPoolFP4.get_key_buffer()` dequantizes the pool regardless, so
  there is no conversion left to save. `ServerArgs` reads the same helper when
  deciding whether the prefill CUDA graph may stay on `tc_piecewise`, so the
  config decision cannot drift from what the kernel will do —
  `TestVarlenAbsorbedCapabilityContract` pins that. The arch half is asserted
  (not merely skipped over) by `TestVarlenAbsorbedArchGate`, which runs on
  `1-gpu-large` (H100, SM 9.0) every PR.

## Production-Unsupported

These combinations are explicitly rejected by the production speculative
multi-step draft backends and cannot ever appear at runtime.

- **FlashInfer MLA tree verify / draft-extend with `topk > 1`** — raised by
  `FlashInferMLAMultiStepDraftBackend.__init__` at
  `python/sglang/srt/layers/attention/flashinfer_mla_backend.py:910-913`:
  `if topk > 1: raise ValueError("Currently Flashinfer MLA only supports topk=1
  for speculative decoding")`. Dispatcher: `draft_utils.py:126-132`.
- **FlashMLA tree verify / draft-extend with `topk > 1`** — raised by
  `FlashMLAMultiStepDraftBackend.__init__` at
  `python/sglang/srt/layers/attention/flashmla_backend.py:555-558`. Dispatcher:
  `draft_utils.py:173-180`.
- **TRT-LLM MLA tree verify / draft-extend with `topk > 1`** —
  `TRTLLMMLAMultiStepDraftBackend` inherits from
  `FlashInferMLAMultiStepDraftBackend` (`trtllm_mla_backend.py:1223-1229`).
- **Tokenspeed MLA tree verify / draft-extend with `topk > 1`** —
  `TokenspeedMLAMultiStepDraftBackend` inherits from
  `TRTLLMMLAMultiStepDraftBackend` (`tokenspeed_mla_backend.py:341-347`).
- **Cutlass MLA extend / verify / draft-extend** — `CutlassMLABackend` only
  overrides `forward_decode` (`cutlass_mla_backend.py:226`) and only handles
  `is_decode_or_idle` in `init_forward_metadata*` (`cutlass_mla_backend.py:86,
  156, 197`). Anything else falls through to FlashInfer MLA.
- **FlashInfer-MLA `DRAFT_EXTEND_V2` graph capture/replay** —
  `flashinfer_mla_backend.py:432,501` only route through `is_draft_extend()`
  (default `include_v2=False`); `else: raise ValueError("Invalid mode")` at
  `flashinfer_mla_backend.py:454-455,512`.
- **All MLA backends fixed page size** — FlashMLA forces `page_size=64`,
  Cutlass MLA forces `page_size=128`, TRT-LLM MLA and Tokenspeed MLA force
  `page_size in {32, 64}`.

## Backend Container Gate (SM10.x)

`test_flashinfer.py::test_runner_mode_eagle_draft_cuda_graph_runner_cases`
skips on `major >= 10`. The FlashInfer MLA multi-step draft backend
(`FlashInferMLAMultiStepDraftBackend`) ships with an SM9x-targeted decode
kernel in the current container; on SM10.x it falls back to a generic path
that doesn't restore metadata buffers correctly under graph replay, producing
~22 abs-diff vs the reference. The eager and DRAFT_EXTEND paths are
unaffected; only this CG decode runner regresses. Update FlashInfer to a
version that ships an SM10.x-compiled MLA multi-step decode kernel to clear.

See `KNOWN_FAILURES.md` §3 for the full root cause + fix.

## Next Work

- Fix or work around the FlashMLA `DRAFT_EXTEND` graph capture path (either
  override capture/replay in `FlashMLABackend` to use its 2D layout, or
  allocate both parent-style 1D and FlashMLA-style 2D buffers and route
  `DRAFT_EXTEND` to the parent path).
- Switch `mla/test_cutlass_mla.py` to `ForwardMode.DECODE` so it actually
  exercises `CutlassMLABackend.forward_decode` instead of falling through to
  FlashInfer MLA when SM 10.0+ is available.
- Add hardware-gated tests for `cutlass_mla`, `trtllm_mla`, and `tokenspeed_mla`
  decode (chain spec only) when the appropriate hardware/KV dtype fixtures are
  available.
