# Kimi-K3 Standalone Kernel Port Design

## Context

SGLang PR #32541 contains the Kimi-K3 Day-0 implementation. Its branch mixes
standalone GPU kernels with model definitions, serving integration, scheduler
changes, disaggregated-serving support, speculative decoding, and other runtime
work. The standalone kernels should land on `main` first so the remaining
Day-0 diff can depend on reviewed, directly tested primitives.

This port starts from `origin/main` at `f4e0ac382e` and takes the final kernel
implementations from `origin/kimi-k3` at `578edb240a`, including the follow-up
kernel fixes and optimizations already merged into that branch.

## Goals

- Port every standalone kernel required by Kimi-K3, including K3-specific
  single-GPU and multi-GPU kernels.
- Keep the port independent of Kimi model classes, scheduler state, server
  arguments, and serving workflows.
- Give every public kernel family a direct correctness test with explicit
  hardware requirements.
- Remove development-only benchmarks and debugging artifacts from the port.
- Organize the result into reviewable commits while keeping one kernel-only
  prerequisite branch for the Day-0 PR.

## Non-Goals

- Do not wire the kernels into `KimiK3ForCausalLM` or any serving path.
- Do not port Kimi parsers, model configuration, multimodal processors,
  speculative-decoding orchestration, disaggregated serving, or scheduler
  changes.
- Do not introduce performance changes beyond the final implementation already
  present on the Kimi-K3 branch, except where decoupling or testability requires
  a behavior-preserving refactor.
- Do not retain one-off tuning, profiling, or benchmark scripts in the
  repository.

## Port Structure

The branch will use four logical implementation commits.

### 1. Shared JIT Infrastructure and Generic Kernels

Port generic headers, JIT compiler support, and reusable kernels such as:

- MLA concatenate/cache operations
- fused vision RoPE
- `add3`
- tiny BF16 GEMM
- MoE alignment, routing, top-k, route-and-quant primitives
- image preprocessing and sampling fallbacks where they are standalone

Only generic support code strictly required to compile or call these kernels is
allowed outside `python/sglang/kernels/`.

### 2. KDA Kernels

Port the complete KDA kernel family:

- fused and packed decode
- ReplaySSM speculative-decode state handling
- CuTeDSL and PTX prefill implementations
- KDA MTP decode
- associated state-scatter and cache-index primitives

The wrappers must expose tensor-level APIs and must not depend on model or
scheduler objects.

### 3. Kimi-K3 Fused Compute Kernels

Port K3-specific compute primitives under
`python/sglang/kernels/ops/kimi_k3/`, including:

- SiTU and masked post-quant activation
- MoE front-end fusion
- MLA output gate
- attention-residual fusion
- K3-specific tiny-GEMM dispatch

Shape-specific dispatch tables and generated tuning configuration files remain
with the kernel because they are part of its execution contract.

### 4. Kimi-K3 Distributed Kernels

Port:

- fused all-reduce
- GEMM + all-gather
- GEMM + all-reduce
- sequence-parallel collectives
- persistent symmetric-buffer support required by these kernels

The public APIs accept explicit tensors and communicator state. Tests may build
process groups and communicator fixtures, but must not initialize a model,
engine, scheduler, or `ServerArgs`.

## Dependency Rules

- A kernel may depend on PyTorch, Triton, CuTeDSL/CUTLASS DSL, CUDA bindings,
  and existing SGLang kernel/JIT helpers.
- A kernel may use an existing generic custom-op registration helper.
- New dependencies on `sglang.srt.models`, scheduler components, server
  arguments, or Kimi runtime layers are prohibited.
- If a copied test currently needs runtime wiring, rewrite the fixture around
  the direct kernel API. If that cannot be done without porting runtime
  behavior, omit that test and add a direct correctness test instead.
- Imports must remain lazy where optional architecture-specific toolchains are
  unavailable.

## Test Design

Tests follow `test/README.md` and the SGLang CI registration contract.

Each kernel family will have:

- a reference implementation using PyTorch or an existing trusted kernel;
- parameterized production shapes plus at least one small diagnostic shape;
- dtype and tolerance declarations;
- boundary coverage for supported token counts, padding, masks, or topology;
- explicit architecture and GPU-count gates;
- deterministic seeds and isolated output buffers;
- CUDA graph replay coverage where graph safety is part of the kernel contract.

Single-GPU kernels use the lightest compatible kernel suite. Multi-GPU tests
use the minimum required world size and compare against PyTorch distributed or
an unfused composition. Tests that require GB/B300 topology remain explicitly
gated rather than silently passing on unsupported hardware.

The existing `test_symm_buffers.py` pattern is not acceptable because it
depends on K3 runtime layers and `ServerArgs`; it will be replaced by a direct
communicator/buffer fixture.

## Benchmark Cleanup

No development-only benchmark introduced by the Kimi-K3 branch will be ported.
This includes one-off scripts under:

- `benchmark/bench_linear_attention/`
- `benchmark/hicache/`
- `benchmark/kernels/kimi_k3/`
- `test/registered/jit/benchmark/`
- `test/registered/kernels/benchmark/`

Performance validation will be run from reproducible commands during
development and summarized in the PR description. A benchmark may remain only
if it is already part of an established SGLang CI benchmark contract and
provides a stable regression threshold; none is assumed necessary initially.

## Validation

Validation proceeds in increasing cost:

1. diff audit ensuring no model/server/scheduler integration leaked in;
2. formatting, import, compile, and CPU-collectable test checks;
3. single-GPU correctness tests on NVIDIA B300;
4. architecture-specific KDA and fused-compute tests;
5. 4-GPU and 8-GPU distributed-kernel correctness tests on B300;
6. targeted Hopper validation for kernels with a distinct SM90 path;
7. comparison of the standalone-kernel results with the final Kimi-K3 branch
   for representative production shapes.

Failures caused by unsupported hardware must skip with a precise reason.
Compilation errors, wrong results, leaked distributed state, and CUDA graph
replay failures are hard failures.

## Deliverables

- branch `bbuf/kimi-k3-standalone-kernels` based on current `origin/main`;
- four reviewable implementation commits plus test-only cleanup commits when
  useful;
- a kernel/test coverage inventory;
- local and remote validation logs;
- a final kernel-only diff suitable for opening as a prerequisite PR to
  `main`.
