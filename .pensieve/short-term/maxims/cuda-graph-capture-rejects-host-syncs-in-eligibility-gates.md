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
