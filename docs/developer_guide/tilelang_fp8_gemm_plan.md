# TileLang FP8 GEMM Modernization Plan

This document tracks the plan for reintroducing the TileLang FP8 W8A8 blockwise
GEMM backend on top of current main.

## Background

The historical branch `remotes/cscyuge/tilelang_w8a8_blockwise` added a
TileLang backend for FP8 W8A8 blockwise linear layers. That branch was built
against an older SGLang codebase and used a custom tuning stack with per-device
JSON config files.

Current main has moved on in several relevant areas:

- FP8 GEMM dispatch now includes DeepGEMM, FlashInfer TRTLLM, FlashInfer
  CUTLASS, FlashInfer DeepGEMM, CUTLASS, Triton, and AITER paths.
- MXFP8 has separate dispatch behavior and must not be accidentally routed
  through the blockwise FP8 path.
- TileLang already exists in the tree for NSA attention, but not for FP8 GEMM.
- Newer TileLang versions provide built-in autotuning and disk cache support,
  so the old Ray tuner and static JSON config-first design should not be
  carried forward as the primary implementation.

## Decisions

- Use TileLang `>=0.1.9` as the target version for the initial implementation.
- Treat TileLang as an optional extra and documented manual dependency, not as
  a hard dependency for all SGLang installs.
- Make explicit `--fp8-gemm-backend tilelang` fail fast on unsupported
  environments, shapes, dtypes, or scale layouts. It should not silently
  fallback to another backend.
- Limit the initial support and validation scope to SM89 and SM90 GPUs because
  those are the machines currently available for testing.
- Support exporting selected best configs for reproducible CI benchmarks, in
  addition to using TileLang's normal cache.

## Known Issues From The Old Branch

- `update_tilelang_config()` existed but was not wired into `ModelRunner`, so
  every rank could attempt precompile with default settings.
- The error message referenced `SGLANG_ENABLE_TILELANG_GEMM`, but that
  environment variable did not exist.
- Documentation and scripts disagreed on command names such as
  `tune_tilelang_gemm.py` versus `tuning_tilelang_gemm.py`.
- The implementation depended on checked-in per-device JSON tuning files.
- Auto backend selection did not include TileLang, even though the branch
  history suggested an intention to enable it broadly.
- The implementation predated current FP8 dispatch changes and therefore must
  be ported manually instead of rebased as-is.

## Goals

- Add an explicit `--fp8-gemm-backend tilelang` backend for W8A8 blockwise FP8
  linear layers.
- Use TileLang's current autotuning/cache APIs instead of maintaining a custom
  Ray tuning system as the default path.
- Preserve existing FP8 backend behavior unless TileLang is explicitly selected.
- Keep MXFP8 routing separate.
- Compile or autotune kernels before CUDA graph capture when precompile is
  enabled.
- Provide correctness and benchmark coverage against existing backends.
- Export selected best configs so CI benchmarks can be reproduced without
  relying only on an opaque local cache state.

## Non-Goals

- Do not make TileLang the default `auto` FP8 GEMM backend in the first PR.
- Do not remove existing DeepGEMM, FlashInfer, CUTLASS, Triton, or AITER paths.
- Do not require TileLang as a hard dependency for all SGLang installations.
- Do not merge the old static config set as the long-term tuning mechanism.

## Proposed Architecture

### Runtime Integration

Add a new optional package under:

```text
python/sglang/srt/layers/tilelang_gemm_wrapper/
```

The wrapper should expose a small API:

- `is_available() -> bool`
- `update_tilelang_config(gpu_id, server_args) -> None`
- `gemm_nt_f8f8bf16(lhs, rhs, out) -> None`
- `warmup_or_autotune_shapes(shapes) -> None`
- `clear_cache() -> None`

`fp8_utils.py` should only import this wrapper in a way that keeps normal
startup working when TileLang is not installed.

### Dispatch

Add `tilelang` to `FP8_GEMM_RUNNER_BACKEND_CHOICES` and
`Fp8GemmRunnerBackend`.

When `--fp8-gemm-backend tilelang` is selected:

- Validate CUDA device support and TileLang `>=0.1.9` import availability.
- Route only standard W8A8 blockwise FP8 linear calls.
- Keep MXFP8 on its existing dispatch path unless a separate TileLang MXFP8
  backend is explicitly designed later.
- Fail fast with a clear error for unsupported inputs instead of falling back
  silently to Triton or another backend.

Do not include TileLang in `auto` until performance and stability have been
validated on the target GPU matrix.

### Dependency Model

TileLang should remain optional. The initial implementation should document how
to install TileLang manually with `pip install 'tilelang>=0.1.9'` or through the
`sglang[tilelang]` optional extra, but normal SGLang imports and non-TileLang
backends must work when TileLang is absent.

### Autotuning And Cache

Use TileLang's built-in autotuning as the primary tuning path. The wrapper
should:

- Define a compact legal config space for each kernel family.
- Use fixed profiling inputs through TileLang's input supply mechanism where
  dynamic shapes require concrete tensors.
- Cache tuned kernels by `(device, dtype, block_shape, M bucket, N, K,
  kernel_type or layout)`.
- Respect TileLang cache environment variables and avoid inventing a parallel
  cache format unless there is a demonstrated gap.
- Export selected best configs for reproducible CI benchmarks.

The old JSON configs can be used as seed data or benchmark references, but they
should not be required for correctness.

### Precompile

Mirror the DeepGEMM integration pattern:

- Wire `tilelang_gemm_wrapper.update_tilelang_config(gpu_id, server_args)` into
  `ModelRunner` when TileLang is available or selected.
- Only the first local rank should perform expensive precompile/autotune work.
- Disable symmetric memory context during compile/autotune if compilation can
  allocate CUDA buffers.
- Ensure kernels used during CUDA graph capture have already been compiled.

### Shape And Scale Contracts

The initial backend should support:

- `input`: `(M, K)` after flattening batch dimensions.
- `weight`: `(N, K)`, FP8 E4M3.
- `weight_scale`: FP32 block scale layout `(ceil(N / 128), ceil(K / 128))`.
- `input_scale`: `None`.
- `block_size`: `[128, 128]`.
- output dtype: `bfloat16` initially.

Unsupported dtypes, block sizes, shape constraints, or scale layouts should
raise a clear error when TileLang is explicitly selected. Silent fallback is not
allowed for `--fp8-gemm-backend tilelang`.

## Implementation Phases

### Phase 1: Minimal Explicit Backend

- Add backend enum and CLI choice.
- Add TileLang wrapper skeleton with optional import behavior.
- Add explicit dispatch path for `--fp8-gemm-backend tilelang`.
- Add one correct non-autotuned baseline kernel path for `[128, 128]` blockwise
  FP8 GEMM.
- Fail fast for unsupported TileLang inputs.
- Keep TileLang out of `auto`.

### Phase 2: Autotuning Integration

- Replace fixed tuning configs with TileLang autotuning.
- Add config-space generation for base, swapAB, split-K, and split-K swapAB
  variants if all remain useful on current TileLang.
- Add precompile/autotune hooks and cache-key strategy.
- Add developer documentation for tuning and cache behavior.
- Add best-config export support for reproducible CI benchmarks.

### Phase 3: Validation And Benchmarking

- Add standalone correctness tests for representative shapes.
- Add manual benchmark scripts comparing TileLang with Triton, CUTLASS,
  DeepGEMM, and FlashInfer where available.
- Add server-level smoke coverage with an FP8 blockwise model and explicit
  TileLang backend.
- Document tested GPUs and known fallback cases.

### Phase 4: Auto Policy Decision

After Phase 3 data is available, decide whether `auto` should select TileLang
for any specific device and workload class. This decision should be performance
data driven and should not be part of the initial merge.

## Validation Matrix

Minimum correctness matrix:

- GPUs: SM89 and SM90.
- Shapes: decode-style small `M` values, medium batch, and prefill-style large
  `M` values.
- Model dimensions: common Qwen and DeepSeek dense linear shapes.
- Dtypes: `bfloat16` output first.
- Backends for comparison: Triton everywhere, DeepGEMM on Hopper where
  available, FlashInfer/CUTLASS where available.

Minimum serving matrix:

- Launch with `--fp8-gemm-backend tilelang`.
- Verify no JIT/autotune occurs during CUDA graph replay.
- Verify clear fail-fast behavior on unsupported shapes.
- Verify behavior with tensor parallelism greater than one.

## PR Breakdown

1. Backend registration and wrapper skeleton.
2. Correct baseline TileLang kernel and explicit dispatch.
3. Autotuning/cache/precompile integration.
4. Benchmarks, tests, and documentation.
5. Optional auto-selection policy after performance review.

## Open Questions

- What export format should selected best configs use so CI can consume them
  reproducibly?
- Which SM89 and SM90 GPU models are required for signoff?
- After SM89 and SM90 data is available, should TileLang participate in `auto`
  for either architecture?
