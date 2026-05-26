# GDN Attention Capability Matrix

This folder covers GDN hybrid-linear attention with a full-attention backend plus
the Triton GDN linear-attention kernel. Expected outputs use a separate pure
PyTorch gated-delta recurrence reference, not Triton/FLA GDN kernels.

## Current Matrix

| Full-attention backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `torch_native` | Full representative GDN input-shape sweep | PCG/BCG extend | Not enabled | No CUDA graph or speculative coverage. |
| `triton` | Full representative GDN input-shape sweep | CUDA graph decode; PCG/BCG extend | EAGLE chain/tree verify; EAGLE chain/tree CUDA graph replay | Tree verify uses a scoped `5e-2` absolute tolerance for bf16 recurrent accumulation. |
| `flashinfer` | Full representative GDN sweep with 64-dim heads for SM90 prefill constraints | CUDA graph decode; PCG/BCG extend | EAGLE chain/tree verify; EAGLE chain/tree CUDA graph replay | Same scoped tree tolerance as Triton. |

## Input And Config Coverage

- Page size 1, exact-page, crossing-page, ragged page-boundary, page-size-32 crossing, decode boundary, and batch-size-1 decode cases.
- GDN uses speculative Mamba state buffers for target verify coverage.
- The split-op tests verify live-token slicing with a larger static token buffer.

## Current Progress

- Phase 2 eager correctness covers representative GDN input layouts against a
  pure PyTorch recurrence reference.
- Phase 3 coverage exercises CUDA graph decode on Triton and FlashInfer, plus
  paged/chunked and batch-chunked extend paths.
- Phase 4 EAGLE chain/tree coverage includes recurrent state buffers and graph
  replay for the currently stable full-attention backends.

## Production-Unsupported

- **HybridLinearAttnBackend CUDA-graph capture/replay outside
  `DECODE_OR_IDLE` / `TARGET_VERIFY`** — `MambaAttnBackendBase._capture_metadata`
  / `_replay_metadata`
  (`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:493-572`)
  both raise `ValueError(f"Invalid forward mode: {forward_mode=}")` for
  anything other than decode-or-idle and target-verify. This is the underlying
  contract for GDN's `Mamba2AttnBackend` as well as for KDA, Lightning, and
  Mamba2 in this codebase. So `DRAFT_EXTEND` / `DRAFT_EXTEND_V2` CUDA-graph
  capture/replay is structurally unreachable for the GDN linear-attention side.
- **HybridLinearAttnBackend `_forward_metadata` modes** — same file
  (`hybrid_linear_attn_backend.py:246`): non-decode, non-extend modes raise
  `ValueError`. The legal modes are `is_decode_or_idle`, plus
  `is_extend(include_draft_extend_v2=True)` (which subsumes
  `EXTEND` / `MIXED` / `DRAFT_EXTEND` / `DRAFT_EXTEND_V2` / `TARGET_VERIFY` /
  `SPLIT_PREFILL` / `DLLM_EXTEND` per
  `model_executor/forward_batch_info.py:106-115`).

## Next Work

- Add additional linear-attention kernel backend variants when available.
- Consider broader speculative worker tags only after EAGLE chain/tree remains stable across kernels.
