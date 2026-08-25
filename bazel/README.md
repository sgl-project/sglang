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

# Update the small checked Python bootstrap lock.
bazel run //bazel/python:srt_empty_bootstrap_requirements.update

# Smallest real Engine execution path. The current hardware CI image must
# already contain SGLang's CUDA runtime wheels and Qwen3 config metadata.
bazel test --config=cuda //bazel/integration:dummy_model_e2e
```

## Ownership boundaries

| Boundary | Initial Bazel ownership | Current release authority |
| --- | --- | --- |
| Python profiles | `bazel/python/profiles.json` validates manifest/extra selection; each accelerator gets a separate future lock hub | `python/pyproject*.toml` |
| Torch-free SRT bootstrap | `//python/sglang/srt:environ` | setuptools |
| Kernel metadata/dispatch | `//python/sglang/kernels:metadata` | setuptools |
| Kernel JIT sources | `//python/sglang/kernels:jit_sources`; runtime compilation remains intentional | setuptools + Ninja/tvm-ffi |
| AOT CUDA/ROCm/CPU kernels | platform-constrained source and wrapper targets are the next native phase | scikit-build/CMake and platform setup scripts |
| Main Rust extensions | one crate-universe closure from `rust/Cargo.lock`; `sglang-mm` is first | Cargo + setuptools-rust |
| Rust gRPC | separate target over `//proto/sglang/runtime/v1`; generated API parity required before replacing `build.rs` | Cargo/tonic-build |
| Model gateway/router | separate crate universe and Python ABI because its Rust/PyO3/tonic versions differ | Cargo + maturin |
| HF config/tokenizer/download | CPU/network-capable Python targets, separate from weight tensor loading | setuptools |
| Weight loading/cache daemon | protocol and file I/O split from CUDA IPC daemon; daemon remains accelerator-constrained | setuptools |
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
- Leaf Python targets (`environ`, kernel metadata).
- `sglang-mm` pure Rust library and test.
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

- Run the dummy-weight, tokenizer-free Engine smoke under the existing GPU CI
  runtime (`//bazel/integration:dummy_model_e2e`).
- Add a real-weight Qwen2 0.5B Bazel smoke after model data and native wheels
  are immutable inputs.
- Compare Bazel and existing wheel manifests, Python ABI tags, exported
  symbols, RPATH/`DT_NEEDED`, and runtime behavior.
- Add digest-pinned OCI image/load targets.
- Make a Bazel artifact authoritative only after parity on every supported
  hardware/Python matrix entry.

The current E2E target is intentionally tagged `local` and `manual`: Bazel owns
the target/platform boundary, while the existing CI installer still supplies
torch and native wheels. Removing that transitional dependency is a Phase 2
acceptance criterion, not something hidden behind ambient host discovery.
