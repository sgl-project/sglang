---
title: "SenseNova-U1 NPU batching development plan"
description: "Implementation checklist for homogeneous request batching of SenseNova-U1 on Ascend NPU."
---

This page tracks the implementation of homogeneous request-level batching for
SenseNova-U1. It remains outside the public navigation while the feature is in
development.

## Scope

- Single-device text-to-image inference on Ascend NPU. SenseNova-U1 dynamic
  batching remains disabled on other platforms.
- Different prompts and seeds in one batch.
- Requests in a batch must share resolution, sampling steps, CFG settings, and
  output options.
- `think_mode=false` for batches larger than one.
- Request-level batching through the existing SGLang-Diffusion scheduler.

Image-to-image generation, multi-device execution, denoise-step continuous
batching, and NPU graph capture are outside this change.

## Implementation checklist

- [x] Add regression tests for batched prompts, seeds, padding, and singleton
  compatibility (execution pending).
- [x] Pass merged prompts, per-request seeds, and the real batch size through
  `SenseNovaU1GenerationStage`.
- [x] Add batched tokenization with padding and per-sample prefix lengths.
- [x] Make text and image position indexes batch-aware.
- [x] Mask padded prefix tokens during prefix and denoise attention.
- [x] Build independent condition KV caches and reuse the common unconditional
  prefix cache.
- [x] Generate initial noise from independent per-request RNG streams.
- [x] Enable dynamic batching and add a SenseNova-specific request cost estimate.
- [x] Keep sequential multi-output requests outside merged dynamic batches.
- [x] Cover singleton, batch size two, unequal prompt lengths, CFG on/off, and
  unsupported batched think mode in unit tests (execution pending).
- [x] Add real attention-layer prefix KV and two-step denoise comparisons with
  unequal prefix lengths and CFG on/off (execution pending).
- [x] Add scheduler merge/split ordering, output-count mismatch, and incompatible
  resolution tests (execution pending).
- [ ] Extend equivalence coverage to the full model generation loop.
- [x] Validate batched generation on 910C at 1024x1024 with 5 steps and inspect
  the generated images.
- [x] Benchmark batch sizes one and two at 2048x2048 with 50 steps; record
  throughput, latency, and peak NPU memory.
- [x] Profile the NPU attention path and add the native fused inference
  attention path for padded request batches; validate it on 910C.
- [x] Keep denoise KV buffers in FIA-native BNSD layout and precompute repeated
  timestep/noise embeddings on NPU.
- [x] Add an NPU fused RMSNorm path with an environment-variable fallback and
  validate it on 910C.
- [x] Fuse dense MLP gate/up projections into one matmul followed by NPU SwiGLU,
  preserve checkpoint parameter names, and validate it on 910C.
- [ ] Re-profile one complete denoise step before deciding whether TorchAir graph
  capture is justified.
- [ ] Add serving and benchmark commands to the SenseNova-U1 cookbook after NPU
  validation.

The optimization order follows the 910C traces. A CFG-enabled denoise step
contains 84 attention calls and 588 dense matmuls: 42 transformer layers, two
condition branches, and seven projections per layer. NPU FIA avoids the explicit
padded SDPA mask, fused RMSNorm replaces the decomposed normalization, and dense
MLP gate/up fusion removes 84 matmul launches per step. QKV fusion and a denoise
token-type shortcut were evaluated and removed: they improved end-to-end
throughput by only 0.76% and 0.02%, respectively, while QKV fusion increased the
reported peak reserved memory by about 1.44 GB.

## Acceptance criteria

- Batched requests preserve prompt, seed, and output ordering.
- A sample generated in a batch matches its singleton execution within the
  agreed numerical tolerance.
- Padding tokens do not participate in prefix or denoise attention.
- Batch size one retains its existing behavior.
- A failed or unsupported batch returns a clear error instead of silently using
  incorrect inputs.
- Batch size two improves output throughput by at least 10 percent at a validated
  910C workload, or profiling evidence explains why the path should remain
  correctness-only.

## Validation status and next steps

The serving path completed 910C smoke tests with generated-image inspection.
Four-request tests at 2048x2048, 50 steps, CFG 4 produced the following results:

| Configuration | Duration (s) | Throughput (images/s) | Mean latency (s) | Peak memory (MB) |
|---|---:|---:|---:|---:|
| B1, optimizations disabled | 234.28 | 0.01707 | 146.47 | 35698 |
| B2, optimizations disabled | 234.83 | 0.01703 | 176.06 | 37870 |
| B1, FIA + RMSNorm + MLP | 205.94 | 0.01942 | 128.80 | 35718 |
| B2, FIA + RMSNorm + MLP | 195.02 | 0.02051 | 146.33 | 37850 |

With the final NPU optimizations enabled, B2 improves throughput by 5.60% over
B1. Final B2 improves throughput by 20.13% over the B1 disabled-optimization
control while keeping mean latency effectively unchanged. The disabled control
uses this branch and is not an upstream-main measurement.

Python compilation, focused Ruff checks, and whitespace checks run on the
Windows development host. PyTorch, Transformers, and pytest are unavailable
there, so the tensor regression suite must run in the Linux inference environment.

Run the regression suite in the existing Linux inference environment:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py -q
```

Attention-layer prefix-plus-denoise comparisons and scheduler output-order tests
are present in the regression suite. Full model generation-loop equivalence is a
follow-up; the completed serving smoke tests cover the integrated path.

Dynamic batching is available only on Ascend NPU and is opt-in through the
existing `--batching-max-size` flag (default: 1). For a B2 trial, add
`--batching-max-size 2 --batching-delay-ms 20 --enable-batching-metrics` to the validated single-device
launch command and send at least two concurrent compatible requests. Each request
must ask for one output and disable think mode. Keep resolution, steps, CFG and
other sampling settings identical. Use the same prompts and seeds with batch size
one for the baseline, and verify the scheduler reports an actual merged batch.

On Ascend NPU, SenseNova denoising uses
`npu_fused_infer_attention_score` with a right-aligned prefix cache and per-sample
KV lengths. Set `SGLANG_SENSENOVA_NPU_FIA=0` before starting the server to restore
the SDPA plus explicit padding-mask path for A/B comparison. The native path is
enabled by default on NPU and keeps the reusable KV cache in the FIA-native BNSD
layout. SenseNova also enables `torch_npu.npu_rms_norm` and dense MLP gate/up
projection fusion with `torch_npu.npu_swiglu` on NPU. Set
`SGLANG_SENSENOVA_NPU_FUSED_NORM=0` or `SGLANG_SENSENOVA_NPU_FUSED_MLP=0` before
server startup to isolate each fused operator. MLP weights are packed lazily on
their first NPU forward, so performance runs require a full warmup request. The
checkpoint-facing parameter names are unchanged. CUDA and CPU keep their original
operators, and SenseNova-U1 request batching remains disabled on those platforms.
