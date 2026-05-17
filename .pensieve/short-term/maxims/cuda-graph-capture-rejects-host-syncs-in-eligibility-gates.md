---
id: cuda-graph-capture-rejects-host-syncs-in-eligibility-gates
type: maxim
title: CUDA-graph-captured code paths must not host-sync to choose a branch — the branch decision freezes at capture time
status: active
created: 2026-05-14
updated: 2026-05-14
tags: [cuda-graph, capture, host-sync, dispatch, kernel-launch]
---

# CUDA-graph-captured code paths must not host-sync to choose a branch

## One-line Conclusion
> If a code path runs inside a CUDA graph capture, never use `tensor.item()` / `.cpu()` / `.tolist()` to decide which kernels to launch — the captured graph freezes whichever branch fired at capture time, and `cudaErrorStreamCaptureInvalidated` may abort the capture outright.

## Guidance

- Eligibility gates that branch on **device tensor state** must either:
  1. be evaluated outside capture (e.g. before entering the layer dispatch),
     and the result passed in as a Python-static value, or
  2. run unconditionally — all branches must be Python-equivalent so the
     captured graph behaves correctly at any runtime tensor value.
- Python ifs that branch on **shape**, **dtype**, or **scratch presence**
  are fine — these are Python-static at capture time.
- Pre-allocate all scratch buffers in init paths so capture sees no
  `torch.empty` / `torch.zeros` on the hot path. CUDA-graph runners
  typically run a warmup pass that catches first-call allocations, but
  do not rely on that — it does not survive dtype changes.
- When debugging "cuda graph: True" but wrong-path behavior, suspect
  capture-time branch freeze: at capture, dummy `seq_lens=1` is fed in,
  so any "is sequence long enough?" gate returns False and freezes the
  legacy fallback into the captured graph.
- Symptom of host-sync in capture: `cudaErrorStreamCaptureInvalidated`
  during `Capture cuda graph begin` in the model_runner; server then
  shuts down with `RuntimeError: Capture cuda graph failed`.
- The **allocation form** of this trap: lazy `torch.zeros(...)` /
  `torch.empty(...)` inside the captured region (typically gated on
  `tensor.dtype != expected_dtype` or `tensor.shape[0] < bs`). At init
  time the preallocated tensor matches the common case, the dtype/shape
  check is False, no allocation. At first real call with a different
  dtype the allocation fires — sometimes during warmup (silently
  absorbed by the graph pool), sometimes during capture (fails with
  the same `cudaErrorStreamCaptureInvalidated`). Fix by **fail-loud
  validating the input dtype/shape at the eligibility gate** instead
  of silently reallocating.
- The **JIT-compile form** of this trap: a third-party kernel
  (FlashInfer Triton, sgl_kernel, etc.) JIT-compiles a fresh CUDA
  module on the first call for a given shape. Triton's `load_binary`
  inside capture surfaces as `Triton Error [CUDA]: illegal memory
  access` (NOT `cudaErrorStreamCaptureInvalidated`). Symptom: the
  bench/server crashes during the FIRST captured forward call after
  model load. Fix: before capture starts, issue a warmup call at
  **every** captured-bs in the SGLang ladder (`1, 2, 4, 8, 12, 16,
  24, 32, ...`) so each shape's Triton kernel handle is already
  populated. Warming only the worst-case bs is NOT sufficient —
  Triton specializes per shape, and the first capture at bs=1 still
  triggers a fresh `load_binary` if only bs=32 was warmed.
- The **stream/context form** of this trap (observed but currently
  unfixable from our side): some kernel libraries bind their compiled
  Triton handles to the stream/context they were first compiled on.
  Even after warming at every captured-bs OUTSIDE capture, the first
  call INSIDE capture (different stream / graph-stream context)
  re-invokes `load_binary` and fails. Worked-around by NOT using the
  affected backend under graph capture; documented in
  `selector_backends.py:FLASHINFER_TOPK_MAX` block comment.

## Positive form: stable-pointer scratch writes from outside capture

The mirror image of the host-sync rule is a useful pattern: **host-side
writes to a stable device tensor are allowed OUTSIDE the captured
region**, and the captured graph picks up the new contents on each
replay. SGLang already uses this for `hisparse_coordinator.num_real_reqs.fill_(bs)`.
The DS native pipeline uses the same trick for:
* `_native_req_to_token_indexed` — written once per decode step from
  `ModelRunner.forward` (eager + before-replay) and from
  `CUDAGraphRunner.capture_one_batch_size` (capture-time seed). The
  captured `try_native_sparse_decode` does a Python-static
  `.narrow(0, 0, bs)` view that's recorded at trace time; replay
  reads the stable pointer's current contents.
* FlashInfer selector's `_lengths_buf` / `_row_to_batch_buf` —
  preallocated at construction (outside capture), refreshed in-place
  per call. No per-call `repeat_interleave` allocation.

Rule of thumb: **anything that can be a stable device pointer
populated externally should be**. Allocation inside capture is the
hazard; data refresh through a stable pointer is not.

## Boundaries

- The graph-runner's warmup runs the forward 2× before capture; first-call
  allocations land in those warmup passes, not in capture. This is implicit
  and not enforceable from the dispatched code — preallocate explicitly
  when correctness depends on it.
- Outside captured regions (extend / prefill), host syncs are fine. The
  maxim is scoped to whatever the capture engine traces.
- If the captured graph is per-bs (SGLang `cuda_graph_bs`), Python-static
  branches that depend on bs are fine — a separate graph is captured per
  bs, each with its own frozen branch.

## Context Links
- Based on: [[decisions/2026-05-14-ds-v2-replace-fa3-adaptor-with-native-sparse-decode]]
- Related: [[maxims/reduce-complexity-before-adding-branches]]
