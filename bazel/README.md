# Bazel migration

SGLang is adopting Bazel incrementally. Bazel is initially an additional
build-and-test graph; setuptools, scikit-build/CMake, Cargo, and maturin remain
the release artifact authorities until parity checks prove equivalent wheels
and shared libraries.

The foundation follows useful control-plane patterns from
[ZML](https://github.com/zml/zml): Bzlmod-only dependency resolution, a pinned
Bazel version, hashed Python inputs, explicit accelerator platforms, and
checked locks. SGLang does not copy ZML's PJRT GPU packaging because SGLang
compiles PyTorch CUDA/ROCm extensions and has different ABI requirements.

## Commands

```bash
# CPU-safe Python dependency and import boundaries.
bazel test --config=cpu //bazel/python:torch_free_tests

# The first independent native module (pure Rust, no Python/GPU/protobuf).
bazel test --config=cpu //rust/sglang-mm:rlib_is_single_threaded_test

# All currently hermetic foundation tests.
bazel test --config=cpu //:bazel_smoke

# Compile the current native component set (Rust gRPC/MM and C++ n-gram core).
bazel build --config=cpu //:bazel_components

# Update checked Python locks.
bazel run //bazel/python:srt_empty_bootstrap_requirements.update
bazel run //bazel/python:hf_management_requirements.update
bazel run //bazel/python:runtime_import_requirements.update

# Repository CLI and widest CPU runtime import boundary. Both use Bazel's
# Python toolchain, checked wheels, SGLang sources, and runtime data.
bazel run --config=cpu //bazel/integration:sglang_cli -- version
bazel test --config=cpu //bazel/integration:runtime_import_test

# Main Rust extensions, gateway, and direct Bazel wheel assembly.
bazel build --config=cpu //:bazel_components
bazel test --config=cpu \
  //bazel/rust:sgl_model_gateway_lock_parity_test \
  //sgl-model-gateway:internal_tests
bazel build --config=cpu //sgl-model-gateway:sgl-model-gateway
bazel build --config=cpu \
  --define=SGLANG_WHEEL_VERSION=0.0.0.dev0 \
  //bazel/packaging:main_wheel

# Hardware targets use the matching pre-provisioned PyTorch/toolchain image.
bazel build --config=cpu //python/sglang/kernels/aot:cpu_wheel
bazel build --config=cuda //python/sglang/kernels/aot:cuda_wheel
bazel build --config=rocm //python/sglang/kernels/aot:rocm_wheel
bazel test --config=cuda //bazel/integration:dummy_model_e2e
bazel test --config=cuda //bazel/integration:qwen2_real_weight_e2e
```

## Verified migration state

- The Bazel-owned CPU runtime imports the public package, CLI, Engine class,
  kernel registry, and environment layer from declared runfiles.
- `_grpc`, `_multimodal`, and `_server` compile as Bazel native extensions with
  their expected `PyInit_*` symbols.
- The model gateway rlib, binary, abi3 shared library, lock parity test, tonic
  0.12/0.14 contract test, and policy tests build successfully.
- Native host JIT `ngram_corpus_ffi.so` links against a SHA-pinned TVM-FFI
  wheel.
- CPU and ROCm `sglang-kernel` wheels build and import through Bazel wrapper
  targets. CUDA builds use immutable Bazel-fetched CUTLASS, fmt, Triton,
  FlashInfer, sgl-attn, and FlashMLA sources.
- Dummy-weight Qwen3 and pinned real-weight Qwen2-0.5B Engine tests passed on an
  H200. The Qwen2 test verifies the model revision, weight digest, token IDs,
  decoded text, and completion metadata.
- The transitional PEP 517 wheel and authoritative wheel match at 4,230 paths,
  3,377 imports, and three native modules. The direct Bazel wheel has the same
  manifest, tags, version, native module destinations, and valid RECORD.

## Ownership boundaries

| Boundary | Initial Bazel ownership | Current release authority |
| --- | --- | --- |
| Python profiles | `bazel/python/profiles.json` validates manifest/extra selection; `runtime_import` is the checked Linux x86_64/cp312 CPU import lock; each accelerator gets a separate future lock hub | `python/pyproject*.toml` |
| Torch-free SRT bootstrap | `//python/sglang/srt:environ` | setuptools |
| Kernel metadata/dispatch | `//python/sglang/kernels:metadata` | setuptools |
| Kernel JIT sources | `//python/sglang/kernels:ngram_corpus_ffi.so` compiles and links the host TVM-FFI adapter; device JIT remains intentional | setuptools + Ninja/tvm-ffi |
| AOT CUDA/ROCm/CPU kernels | `cpu_wheel`, `cuda_wheel`, and `rocm_wheel` wrap existing builders; CUDA source downloads are immutable Bazel repositories | scikit-build/CMake and platform setup scripts |
| Main Rust extensions | one crate-universe closure builds `_multimodal`, `_grpc`, `_server`, and the pure Rust libraries | Cargo + setuptools-rust |
| Rust gRPC | `//rust/sglang-grpc:sglang_grpc_core` runs the existing tonic build script with Bazel's declared protoc and proto input | Cargo/tonic-build |
| Model gateway/router | separate 838-package crate universe builds the gateway binary and abi3 shared library while preserving dual tonic versions | Cargo + maturin |
| HF config/tokenizer/download | `//python/sglang/srt/utils/hf_transformers:hub` and `//python/sglang/srt/utils:hf_transformers_patches` are torch-free; model config remains separate | setuptools |
| Weight loading/cache daemon | `//python/sglang/srt/weight_cache:protocol` is torch-free; CUDA IPC transport and daemon remain accelerator-constrained | setuptools |
| Model artifacts | runtime inputs except for a future pinned tiny smoke fixture | Hugging Face/runtime cache |

`srt_empty` is represented as a profile, not as the CPU product. It selects
`runtime_base` from `pyproject_other.toml` and must stay free of packages known
to pull torch or triton. `cpu` selects the separate `pyproject_cpu.toml` product
and may contain torch CPU dependencies.

Python accelerator hubs must remain separate. A Linux x86_64 platform is not
enough to distinguish CPU, CUDA, ROCm, XPU, NPU, or MUSA wheels. Application
targets will consume stable aliases while `select()` chooses the hub associated
with the accelerator constraint.

## Migration phases

### 1. Foundation and independently buildable units

- Bzlmod, Bazel/Python/Rust/uv pins, module and language locks.
- Explicit CPU/CUDA/ROCm/XPU/NPU/MUSA/MPS/HPU constraints.
- CPU-safe packaging contract tests.
- Leaf Python targets (`environ`, kernel metadata, HF Hub/compat, weight-cache
  protocol).
- `sglang-mm` pure Rust library/test and the gRPC PyO3 shared library.
- A real C++ compile/link target for the host n-gram corpus core.
- Additive CPU Bazel CI; no release changes.

Acceptance: lockfiles are unchanged under `--lockfile_mode=error`, tests use no
host Python packages, and a CPU configuration does not select accelerator
wheels or toolchains.

### 2. Component builds

- Split HF management, weight protocols, loader families, and entrypoints into
  explicit Python targets.
- Add native PyO3 targets for `_multimodal`, `_grpc`, and `_server`, with
  import/GLIBC parity against setuptools-rust.
- Add a separate gateway crate universe and abi3 extension target.
- Move Cutlass, fmt, Triton, FlashInfer, sgl-attn, and FlashMLA downloads from
  CMake actions into immutable Bazel repositories while preserving current
  hashes.
- Wrap existing CPU/CUDA/ROCm kernel builders with declared inputs; then migrate
  leaf native kernels only after symbol and numerical parity tests pass.

Acceptance: every boundary has an independently buildable target, build actions
perform no network access, and unsupported hardware combinations fail during
analysis.

### 3. Integration and artifact parity

The runtime, real-model E2E, and wheel-manifest milestones are complete. Bazel
must not become the release authority yet:

- Bazel Rust outputs currently inherit the host GLIBC and can require newer
  symbols than the manylinux-built authoritative extensions. A pinned
  manylinux-compatible Rust/C++ sysroot is required.
- CPU/CUDA/ROCm wheel wrappers still consume the ambient PyTorch compiler ABI
  and toolkit. CUDA source downloads are declared, but the compiler, sysroot,
  Python development files, and PyTorch libraries must become toolchains.
- Release-only `auditwheel repair`, ELF `DT_NEEDED`/RPATH policy, architecture
  matrices, and digest-pinned OCI image targets remain outside Bazel.
- Hardware wheel builds remain shadow artifacts until all ABI and numerical
  tests pass across supported CUDA/ROCm/Python variants.
