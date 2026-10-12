---
name: kernel-organization
description: Apply the SGLang kernels RFC when adding, moving, splitting, or reviewing kernel APIs, registry metadata, kernel tests, benchmarks, and model-specific implementations. Use with add-jit-kernel, add-sgl-kernel, and write-sglang-test for placement and migration checks.
---

# Kernel organization

Read `python/sglang/kernels/README.md` and `test/README.md` before choosing a
location. [RFC #29630](https://github.com/sgl-project/sglang/issues/29630)
establishes the namespace; subsequent migrations #32148 and #40922 clarify
logical grouping and remove the old `_jit_` filename prefix.

## Choose the owner

- Put callable SGLang kernel APIs under `sglang.kernels.ops.<group>`. Runtime
  and integration tests import from that namespace, including its submodules.
  The `sgl_kernel` wheel retains its own public API and packaging tests.
- Group by computation, not model or GPU: GEMM and GEMV in `gemm`, expert
  routing in `moe`, attention index selection in `attention`, sampling in
  `sampling`, normalization in `layernorm`. A quantized GEMM is still a GEMM;
  quantization alone belongs in `quantization`.
- Model-specific files/subpackages and tuning data are allowed **inside** a
  logical group. Do not add a model bundle or an implementation file directly
  under `ops/`. Propose a new logical group only for a distinct responsibility,
  and update `ops._GROUPS`, documentation, and tests together.
- Classify a fused operator by its complete contract. Do not split a fused
  kernel into separate launches just to separate norm, RoPE, or quantization.
  Split unrelated public entry points that happen to share a source file.
- Keep shared CUDA build/runtime infrastructure in `kernels/jit`; operator
  wrappers call it from their logical group. Keep process groups, communicator
  state, model dispatch, and buffer ownership in `srt`. K3-specific adapters in
  `srt/layers/communication/` need not pretend to be generic interfaces.

## Register the public entry point

Add lazy `KernelSpec` metadata in the owning group's `__init__.py`, or use the
existing `BaseFusedOp` registration when the operation has interchangeable
backends. Group imports must not import GPU implementations or compile kernels.

Use `<group>.<name>` for the op id and `module:callable` for the target. Preserve
existing backend dispatch and describe actual device/architecture restrictions
with `CapabilityRequirement`; JIT/AOT are not device types. Input shape/dtype
checks remain part of the entry point's contract. Registering a torch custom op
does not register it in the SGLang kernel inventory. Predicates, private JIT
factories, reference helpers, and runtime classes are not separate kernel APIs.

## Place and preserve tests

- Kernel numerical tests: `test/registered/kernels/ops/<group>/`.
- CI microbenchmarks: `test/registered/kernels/benchmark/<group>/`.
- Inventory/selector tests: `test/registered/unit/kernels/`.
- Runtime unit tests: `test/registered/unit/<subsystem>/`, following the tested
  runtime module. A CUDA allocation does not make a runtime test a kernel test.
- Non-CI smoke scripts and benchmarks: `test/manual/kernels/`; do not leave
  standalone test/benchmark entry points in production operator modules.
- Shared test helpers: `sglang.test.kernels`. Preserve vendored upstream trees
  and AOT wheel packaging boundaries instead of reorganizing them incidentally.

For a move/split, preserve assertions, parametrization, fixtures, platform skips,
execution entry points, and CI stages/runners. Apportion existing time estimates
across split files; do not duplicate the original budget for every output file
or silently drop a registration. Follow `write-sglang-test` for CI registration.

## Verify a migration

1. Search all runtime, test, benchmark, documentation, patch-string, and lazy
   registry references before deleting the old path. Check relative imports and
   package re-exports as well as direct imports. Do not leave forwarding shims.
2. Keep tuning files with the GEMM that loads them and preserve relative lookup
   behavior. Check wheel/package inclusion as well as source-tree execution.
3. Preserve module singleton state: every runtime caller must import the same
   new communication adapter, not a second copy of its buffer registry.
4. Separate relocation commits from semantic changes. Use
   `mechanical-refactor-verify` to reproduce moves, and compare test inventories
   and CI registration coverage before/after. Only delete an experimental path
   after checking its call sites, flags, source, and dedicated tests together.
5. Run the namespace/dispatch CPU tests, the registered-test validation hook,
   pre-commit, and relevant GPU tests when available. State exactly which GPU
   checks ran; import/AST checks do not prove numerical or performance parity.

Do not infer violations solely from a model name inside a group, a missing
`_jit_` prefix, use of a runtime utility, or absence of `BaseFusedOp` inheritance.
