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
```

## Ownership boundaries

| Boundary | Initial Bazel ownership | Current release authority |
| --- | --- | --- |
| Python profiles | `bazel/python/profiles.json` validates manifest/extra selection; `runtime_import` is the checked Linux x86_64/cp312 CPU import lock; each accelerator gets a separate future lock hub | `python/pyproject*.toml` |
| Torch-free SRT bootstrap | `//python/sglang/srt:environ` | setuptools |
| Kernel metadata/dispatch | `//python/sglang/kernels:metadata` | setuptools |
| Kernel JIT sources | `//python/sglang/kernels:ngram_corpus_core` compiles the host C++ core; device JIT remains intentional | setuptools + Ninja/tvm-ffi |
| AOT CUDA/ROCm/CPU kernels | platform-constrained source and wrapper targets are the next native phase | scikit-build/CMake and platform setup scripts |
| Main Rust extensions | one crate-universe closure from `rust/Cargo.lock`; `sglang-mm` is first | Cargo + setuptools-rust |
| Rust gRPC | `//rust/sglang-grpc:sglang_grpc_core` runs the existing tonic build script with Bazel's declared protoc and proto input | Cargo/tonic-build |
| Model gateway/router | separate crate universe and Python ABI because its Rust/PyO3/tonic versions differ | Cargo + maturin |
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

- Extend `//bazel/integration:runtime_import_test` to the dummy-weight,
  tokenizer-free Engine smoke after the CUDA wheel closure is declared.
- Add a real-weight Qwen2 0.5B Bazel smoke after model data and native wheels
  are immutable inputs.
- Compare Bazel and existing wheel manifests, Python ABI tags, exported
  symbols, RPATH/`DT_NEEDED`, and runtime behavior.
- Add digest-pinned OCI image/load targets.
- Make a Bazel artifact authoritative only after parity on every supported
  hardware/Python matrix entry.

`runtime_import` pins cp312/manylinux CPU Torch, TorchVision, and Triton wheels
by URL and digest, then imports the real `Engine` class. A dummy Engine startup
probe reaches `sglang.kernels.ops.kvcache.cache_move` and stops precisely at
`sgl_kernel.copy_all_layer_kv_cache_cpu`: the CPU `sgl_kernel` extension is
built by the platform wheel and is not available as an independent CPU wheel.

A CUDA Engine target therefore needs a separate Linux/Python/CUDA hub. The
current CUDA manifest's accelerator-native closure includes `torch==2.13.0`,
`sglang-kernel==0.4.6.post1`, `flashinfer-python[cu13]==0.6.17`,
`flash-attn-4>=4.0.0b18`, `humming-kernels[cu13]==0.1.12`,
`quack-kernels==0.6.4`, `sgl-deep-gemm==0.1.5.post3`,
`sgl-deep-ep==0.1.2`, `cuda-python>=13.0`, `cuda-tile==1.6.0rc5`,
`nvidia-cutlass-dsl[cu13]==4.6.2`, `nvidia-mathdx==25.6.0`,
`tilelang==0.1.12`, `tokenspeed-mla==0.1.8`, and
`torch-memory-saver>=0.0.9.post1`. The tiny model configuration must also be a
declared runtime artifact. Using installed site-packages or a mutable Hugging
Face cache would hide those ABI and data boundaries.
