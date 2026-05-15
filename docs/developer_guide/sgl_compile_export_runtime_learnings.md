# SGL Compile Export Runtime Learnings

This note captures the practical lessons from moving `@sgl_compile` from a
CUDA-specific ONNX demo into a reusable exported-graph runtime layer.

## Core Learning

The platform should not own ONNX Runtime details. A platform should decide
policy: compile, noop, or export. The export layer should own artifact naming,
metadata, shape policy, conversion, loading, and runtime invocation.

That split gives ONNX-only platforms the same path as CUDA:

```text
@sgl_compile callsite
  -> SRTPlatform policy
  -> ExportArtifactSpec
  -> torch.export.ExportedProgram
  -> ExportRuntime
  -> scheduler/model path callable
```

`CudaOnnxPlatform` is now a thin adapter around `OnnxExportRuntime`. The
`OnnxPlatform` demo shows the same pattern without CUDA graph support, which is
the safer default for vendors whose runtime only supports ONNX graphs.

## Scheduler Shape Behavior

Exported callsites run inside the existing scheduler and sampling path. There is
no separate exported-graph scheduler queue. This means exported runtimes must
accept the shapes produced by normal SGLang batching.

The important shape lesson came from `apply_scaling_penalties`:

- The first request can export with batch size 1.
- Later scheduler batches can call the same exported graph with batch size 3 or
  4.
- CUDA graph replay pads model forward internally, but sampling sees the real
  runtime batch after slicing.

The fix was to make dynamic exports infer tensor dimensions and promote batch-1
examples during export so PyTorch does not specialize the batch dimension to a
constant.

## Artifact Lifecycle

Artifact handling needs explicit modes:

- `build_if_missing`: local development and smoke testing.
- `export_only`: cloud artifact builders that should not change serving
  behavior.
- `load_only`: production serving with prebuilt artifacts only.

Metadata is not optional. It records the graph key, format, shape policy, input
schema, SGLang/PyTorch versions, and mutation contract. This is what lets a
serving runtime reject an incompatible prebuilt graph early instead of failing
inside the scheduler.

## Runtime Performance

The first ONNX implementation copied tensors through CPU/NumPy:

```text
CUDA tensor -> CPU NumPy -> ONNX Runtime -> CUDA tensor
```

That was slower for Qwen3-14B because the exported graph was a small
sampling-side penalty kernel, not the full model. After moving execution into
`OnnxExportRuntime`, CUDA I/O binding could write directly into the in-place
output tensor for `copy_output_to_arg_index` callsites.

The rerun showed parity for the tested workload:

| Mode | Seq tok/s | C4 tok/s |
| --- | ---: | ---: |
| CUDA | `90.9` | `348.4` |
| `cuda_onnx` | `90.8` | `355.2` |

This does not prove full-model ONNX performance. It proves the exported
callsite path can be scheduled and executed without the previous CPU copy tax
for in-place outputs.

## What To Watch Next

- Extend I/O binding beyond in-place single-output callsites by deriving output
  allocation shapes from ONNX metadata or runtime shape inference.
- Add richer shape policy declarations: bounded dims, static dims, and named
  symbolic dims per input.
- Keep ONNX-only platforms conservative about CUDA graph or equivalent graph
  capture support until their runtime can safely participate.
- Migrate only kernels large enough to amortize runtime overhead. Tiny kernels
  are useful for proving the path, but not always useful for performance.
- Keep deterministic sampling in mind: some sampler code still has direct
  `torch.compile` use and is not yet routed through `@sgl_compile`.

## Reproduction Artifacts

Local artifacts from the H100 validation run:

- `/tmp/qwen14_cuda_runtime_abstraction_bench.json`
- `/tmp/qwen14_cuda_onnx_runtime_abstraction_bench.json`
- `/tmp/sglang-qwen14-onnx-runtime-abstraction/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.onnx`
- `/tmp/sglang-qwen14-onnx-runtime-abstraction/sglang.srt.sampling.penaltylib.repetition_penalty.apply_scaling_penalties.metadata.json`

The generated artifacts are intentionally not checked in.
