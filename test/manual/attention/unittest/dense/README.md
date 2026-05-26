# Dense Attention Capability Matrix

This folder covers standard dense MHA/GQA/MQA attention through `RadixAttention`.
Expected outputs come from independent HF-style PyTorch reference modules with
copied random projection weights, not from another SGLang attention backend.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `torch_native` | Full dense input-shape sweep plus MHA/GQA/MQA configs | Eager decode/extend runner checks | Not enabled | No CUDA-graph or replay path: `TorchNativeAttnBackend` does not override `init_forward_metadata_capture_cuda_graph` / `_replay_cuda_graph`, so graph runners are production-unsupported. `TARGET_VERIFY` probe currently fails in torch-native SDPA extend metadata handling. |
| `triton` | Full dense input-shape sweep plus MHA/GQA/MQA configs | CUDA graph decode for MHA/GQA/MQA; PCG/BCG extend for MHA/GQA | EAGLE/Frozen-KV-MTP/DFlash/NGRAM verify; verify CUDA graph for EAGLE/DFlash/NGRAM; production `EAGLEDraftCudaGraphRunner` chain/tree replay; production EAGLE `DRAFT_EXTEND_V2` graph replay | `DRAFT_EXTEND` is intentionally blocked by a reference mismatch. |
| `flashinfer` | Full dense sweep with `head_dim=64` for SM90 prefill constraints | CUDA graph decode for MHA/GQA/MQA; PCG/BCG extend for MHA/GQA | EAGLE/Frozen-KV-MTP/DFlash/NGRAM verify; verify CUDA graph for EAGLE/Frozen-KV-MTP/DFlash; production `EAGLEDraftCudaGraphRunner` chain/tree replay; production EAGLE `DRAFT_EXTEND`; production Frozen-KV MTP draft decode | `DRAFT_EXTEND_V2` graph capture is production-unsupported (see below). |
| `fa3` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Hardware-gated to SM 80-90 for non-MLA, SM 90 only for MLA (`attention_registry.py:172-180`). Decode/speculative CUDA graph replay currently mismatches the PyTorch reference. |
| `fa4` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Effective hardware gate is SM 100+ (only configured as a default on SM100 paths in `server_args.py`). Same current graph blocker as `fa3`. |
| `flex_attention` | Full dense input-shape sweep plus MHA/GQA/MQA configs | PCG/BCG extend | Not enabled | Backend mask callback compatibility is covered; rejects cross-attention / encoder-only (`torch_flex_backend.py:267-270`) and non-causal mode (`torch_flex_backend.py:151`). No CUDA-graph capture path; graph/spec coverage is structurally production-unsupported. |
| `trtllm_mha` | Decode-only MHA/GQA/MQA and page-size-32 coverage | Not enabled | Not enabled | Local SM90 decode passes; prefill reports unsupported architecture and graph replay mismatches. Speculative paths only support `topk == 1` (production note at `server_args.py:2391-2392`; replay branches at `trtllm_mha_backend.py:459,492` explicitly comment "Here we only support topk = 1 for now"). Page size restricted to `{16, 32, 64}` (`server_args.py:2849-2853`). |

## Input And Config Coverage

- Page size 1, page size 16, and representative page size 32.
- Zero-prefix exact page, prefix exact page, total exact page, and page-boundary crossing.
- Ragged batches with lengths below/equal/above a page.
- Decode page-boundary batches and batch-size-1 decode.
- Attention config coverage for MHA, GQA, and MQA is separate from input-layout coverage.

## Current Progress

- Phase 2 eager correctness covers all locally runnable dense backends with an
  independent PyTorch reference.
- Phase 3 runner coverage includes CUDA graph decode where supported plus
  paged/chunked and batch-chunked extend paths.
- Phase 4 now covers synthetic verify runners and production draft-runner graph
  replay for the Triton and FlashInfer paths whose metadata matches the reference.

## Production-Unsupported

- **`torch_native` CUDA graph / speculative** — `TorchNativeAttnBackend`
  (`python/sglang/srt/layers/attention/torch_native_backend.py`) does not
  implement `init_cuda_graph_state` / `init_forward_metadata_capture_cuda_graph` /
  `init_forward_metadata_replay_cuda_graph`. The base class raises
  `NotImplementedError`
  (`python/sglang/srt/layers/attention/base_attn_backend.py:24-55`), so any
  graph-runner / speculative-graph integration that needs these methods cannot
  be exercised.
- **`flex_attention` CUDA graph capture/replay** — same as `torch_native`:
  `TorchFlexAttnBackend` (`python/sglang/srt/layers/attention/torch_flex_backend.py`)
  does not implement the cuda-graph capture/replay hooks. It also explicitly
  rejects non-causal (`torch_flex_backend.py:151`) and cross / encoder-only
  attention (`torch_flex_backend.py:267-270`).
- **FlashInfer non-MLA `DRAFT_EXTEND_V2` graph capture/replay** — both
  `init_forward_metadata_capture_cuda_graph` and
  `init_forward_metadata_replay_cuda_graph` only route through
  `is_draft_extend()` (default `include_v2=False`) at
  `python/sglang/srt/layers/attention/flashinfer_backend.py:651,748`. Any
  `DRAFT_EXTEND_V2` mode falls into the `else: raise ValueError` branches.
- **FlashInfer MLA `DRAFT_EXTEND_V2` graph capture/replay** — same shape, with
  `is_draft_extend()` branches at
  `python/sglang/srt/layers/attention/flashinfer_mla_backend.py:432,501` and a
  final `raise ValueError("Invalid mode")` for anything else
  (`flashinfer_mla_backend.py:454-455,512`).
- **`trtllm_mha` speculative `topk > 1`** — the multi-step draft backend
  `TRTLLMHAAttnMultiStepDraftBackend`
  (`python/sglang/srt/layers/attention/trtllm_mha_backend.py:883-897`) inherits
  from `FlashInferMultiStepDraftBackend`, which does *not* reject topk>1, but
  the replay path itself only handles topk=1 — the comment "Here we only
  support topk = 1 for now" at `trtllm_mha_backend.py:459,492` is load-bearing
  and matches the server-side note that "trtllm_mha requires
  speculative_eagle_topk == 1" (`server_args.py:2391-2392`).

## Next Work

- Debug torch-native target-verify extend metadata.
- Debug Triton `DRAFT_EXTEND` metadata/reference mismatch.
- Debug FA3/FA4 CUDA graph replay and speculative graph mismatches.
- Add backend-specific graph coverage for `trtllm_mha` once local hardware and metadata behavior allow it.
