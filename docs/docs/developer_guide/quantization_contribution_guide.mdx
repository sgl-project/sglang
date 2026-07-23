---
title: "Quantization Contribution Guide"
metatags:
    description: "Guidance for contributing SGLang quantization methods, schemes, backend kernels, tests, and validation results."
---

This guide describes how to add or refactor quantization support in SGLang. It focuses on the common structure used by weight-only and weight-activation quantization methods such as AWQ, GPTQ, compressed-tensors, ModelSlim, Quark, and related backend kernels.

## Design Goals

Quantization code should keep the quantization format semantics separate from hardware-specific execution. This makes it easier to add new formats, reuse kernels across formats, and review platform-specific changes independently.

Follow the architecture proposed in [Quantization Modifications](https://github.com/sgl-project/sglang/issues/15194):

- **Config**: parses model and runtime quantization parameters, validates supported options, and selects the proper scheme.
- **Scheme**: owns quantized weight creation, weight loading, post-processing, and quantized layer wiring for Linear, MoE, embedding, or other module types.
- **Backend kernel**: wraps hardware-specific execution, layout conversion, kernel selection, and kernel calls for GPU (CUDA/HIP/XPU), NPU, or other backends.

Avoid putting config parsing, weight loading, and backend kernel calls in a single monolithic file. If a method needs multiple formats or backends, add a package under `python/sglang/srt/layers/quantization/<method>/` and split schemes into `schemes/`.

## Recommended File Layout

Use this layout for a quantization method that has multiple schemes or backend-specific execution paths:

```text
python/sglang/srt/layers/quantization/<method>/
  __init__.py
  <method>.py
  schemes/
    __init__.py
    <method>_scheme.py
    <method>_linear.py
    <method>_moe.py
    <method>_<variant>.py
```

Backend kernels should live under the hardware backend they target:

```text
python/sglang/srt/hardware_backend/gpu/quantization/<method>_kernels.py
python/sglang/srt/hardware_backend/npu/quantization/<method>_kernels.py
```

Keep shared method selection in the quantization package and keep backend imports narrow. This prevents circular imports and keeps non-target platforms from importing unavailable kernel dependencies.

## Adding or Refactoring a Quantization Method

1. Define the config entry point and register it through `python/sglang/srt/layers/quantization/__init__.py` when needed.
2. Add explicit scheme selection helpers such as `get_linear_scheme` and `get_moe_scheme`.
3. Move layer-specific weight creation and weight loading into scheme classes.
4. Move GPU (CUDA/HIP/XPU), NPU, or other hardware kernel calls into backend kernel modules.
5. Keep Linear, MoE, embedding, and non-linear module handling explicit. Do not assign a Linear quantization method to a module type that needs different semantics.
6. Preserve compatibility for existing quantized checkpoints and runtime flags.
7. Add tests that cover both config parsing and execution paths touched by the change.

For examples, see the AWQ and GPTQ refactors:

- [PR #21126](https://github.com/sgl-project/sglang/pull/21126): splits AWQ schemes, weight initialization, and backend kernel calls.
- [PR #26402](https://github.com/sgl-project/sglang/pull/26402): applies the same scheme/kernel split to GPTQ.

## Tests and Validation

Quantization changes can affect both accuracy and performance. Include validation that matches the blast radius of the change.

For Python-only structure changes:

```bash
ruff check <changed-python-files>
git diff --check
```

For quantized model behavior:

- Launch at least one representative model for each touched quantization method.
- Send a `/generate` request and confirm the output path succeeds.
- Run an accuracy sanity test if the change can affect numerics.
- Include warmup-aware benchmark results when the change affects kernel calls, layout conversion, or dispatch.

For backend-specific changes:

- Validate GPU changes on a supported GPU environment (NVIDIA, AMD, or Intel).
- Validate NPU changes on a supported Ascend environment.
- Include the exact model, quantization flag, backend flag, hardware, and command used in the PR description.

## PR Checklist

Before requesting review, make sure the PR description includes:

- The quantization method and backend paths changed.
- The issue, design proposal, or roadmap item the PR follows.
- Any compatibility notes for existing checkpoints or flags.
- Accuracy results when model outputs can change.
- Benchmark or profiling results when runtime performance can change.
- The exact local checks and model launch tests that were run.

Use the general [Contribution Guide](./contribution_guide) for source setup, formatting, unit tests, CI triggering, and review process details.
