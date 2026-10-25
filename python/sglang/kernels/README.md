# `sglang.kernels` — unified kernel namespace

This package is the public in-tree import surface for callable kernels, per
[RFC #29630](https://github.com/sgl-project/sglang/issues/29630).

```python
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.kvcache import reshape_and_cache_flash
```

## Layout

```
sglang/kernels/
  spec.py        # KernelSpec, KernelBackend, CapabilityRequirement,
                 # PlatformInfo
  registry.py    # process-wide KernelRegistry + register_kernel()
  selector.py    # heuristic select_kernel() and cached get_kernel()
  fused_op.py    # BaseFusedOp: per-operator multi-backend contract
  kernel_api_logging.py  # kernel API crash-debugging decorator
  numerics.py    # shared numerics helpers for bit-exact Triton kernels
  ops/
    <group>/     # one subpackage per operator group (see list below)
  jit/           # shared JIT build/runtime infra: utils/ (incl. the Helion
                 # runtime helper), csrc/, include/, __main__ (KERNEL_PATH)
  aot/           # source tree of the AOT `sgl_kernel` wheel (own API and
                 # packaging; not shipped in the sglang wheel)
  kda_kernels/   # agent-generated kernel backend (Kernel Design Agents)
  cake_kernels/  # agent-generated kernel backend (Cake, shipped by FlashInfer)
  plugin_kernels/  # plugin kernel backend: contracts of external kernel
                   # libraries, as <hardware>/<model>/
```

Operator groups (all populated): `activation`, `attention`, `communication`,
`diffusion`, `elementwise`, `embeddings`, `gemm`, `grammar`, `kv_canary`,
`kvcache`, `layernorm`, `lora`, `lplb`, `mamba`, `memory`, `mm`, `moe`,
`quantization`, `sampling`, `speculative`.

Place model-specific implementations and tuning data inside the corresponding
logical operator group, rather than creating a model-specific top-level group
under `ops/` (a model-named file or subpackage inside a group is fine).

`lora` is its own group because it has a distinct responsibility: the adapter
kernels that only the LoRA runtime in `srt/lora` calls (`dense/` SGMV GEMMs and
their tuning data, `moe/` fused MoE-LoRA). It also keeps the experimental
TRT-LLM LoRA path whole as `dense/trtllm_lora_temp/` and
`moe/trtllm_lora_temp/`; the latter includes the FlashInfer overlay and the
flag-gated routing kernels that `srt/layers/moe/topk.py` reaches.

As of the RFC #29630 finale (#32072) the legacy `sglang.jit_kernel` package has
been **removed**: its shared build/runtime infra moved to `sglang.kernels.jit`
and each JIT-backed operator into its group as
`sglang.kernels.ops.<group>.<op>` (the old `_jit_` filename prefix was removed
in #32148). Tests and benchmarks live under
`test/registered/kernels/` (`ops/<group>/` for tests, `benchmark/<group>/` for
benchmarks); shared test helpers are in `sglang.test.kernels`. Standalone scripts
that are not CI-registered belong in `test/manual/kernels/`, outside the shipped
operator package. Keep vendored upstream trees under their existing maintenance
policy.

Classify an operator by its complete public computation, not by the model
that first used it, and keep a fused operation intact:

- **Collectives win.** Anything that performs a cross-rank collective or
  reads/writes peer memory (all-reduce, all-gather, reduce-scatter, all-to-all,
  multicast/NVLS) belongs in `communication`, whatever compute it fuses (GEMM,
  norm, MoE finalize).
- **Producer + quantization stays with the producer.** Norm + quant is
  `layernorm`, activation + quant is `activation`, a GEMM with a quantizing
  epilogue is `gemm`, MoE dispatch/permute + quant is `moe`. `quantization`
  holds only pure format conversion: quant, dequant, scale-layout conversion,
  repack. A quantized GEMM is still a GEMM.
- **RoPE stays in `attention`**: standalone RoPE, position ids, and QK-norm +
  RoPE fusions.
- **`kvcache` owns KV-cache contents**: ops that read or write the cache
  (store, gather, page copy/move, host transfer, quantize-into-cache, layout
  conversion, clearing cache/state rows). **`attention` owns the metadata**:
  page tables, `cu_seqlens`, `kv_indices`, split/schedule tables, and attention
  index selection. A fused compute + store (e.g. norm + RoPE + quant + store)
  stays with its compute group.
- Otherwise: expert routing is `moe`, matrix multiplication (and its tuning
  data) is `gemm`, normalization is `layernorm`.

Runtime integration that owns process groups, symmetric buffers, workspace
classes, or model dispatch stays in `srt`: the K3, Muse Spark and SP
collective-GEMM communication adapters and workspaces live in
`srt/layers/communication/`, while their callable kernels live in
`kernels/ops/communication/` and take that state as arguments. Likewise the
integration of kernel plugins lives in `srt/kernel_plugins/`, and their
contracts in `kernels/plugin_kernels/` (see below). Reference/test oracles belong in
`sglang.test.kernels`, not in an operator module. Tests of runtime state belong
with the runtime subsystem, even when they need a GPU. Kernel inventory/dispatch
tests belong in `test/registered/unit/kernels/`.

When moving an operator, update its callers, lazy `KernelSpec` targets and op
ids, tests, benchmark imports, and tuning-data paths together. Preserve CI
stage/runner registrations and test cases when splitting files. See the
[kernel organization skill](../../../.claude/skills/kernel-organization/SKILL.md)
for the review checklist.

## Agent-generated kernel backends: `kda_kernels/` and `cake_kernels/`

These two directories hold implementations produced by kernel-generating agents
and will keep growing; they are implementation homes, not operator groups.

- `kda_kernels/` — kernels produced or extended by the Kernel Design Agents
  (Humanize2) workflow, in CUDA, Triton, or CuTe DSL, with their JIT sources in
  `kda_kernels/csrc/`. `KernelBackend.KDA` records that provenance.
- `cake_kernels/` — thin adapters to Cake-generated kernels that FlashInfer
  distributes (generated sources and caches stay in FlashInfer).
  `KernelBackend.CAKE` records that provenance.

Their public ops are registered into the logical groups with
`KernelSpec(op="<group>.<name>", backend=..., target="sglang.kernels.kda_kernels.<mod>:<fn>")`
(or a `cake_kernels` target), and runtime code imports the `ops.<group>`
facade, never these packages. Put a new agent-generated kernel in the matching
directory and register it with that directory's backend (`KDA` or `CAKE`),
whatever language it is written in (see `kda_kernels/README.md`);
`test_kernels_namespace.py` enforces the pairing. List them from the registry
by backend:

```python
import sglang.kernels.ops  # populate the registry
from sglang.kernels import KernelBackend, registry

agent_kernels = [
    spec
    for spec in registry.all_specs()
    if spec.backend in (KernelBackend.KDA, KernelBackend.CAKE)
]
```

## Plugin kernel backend: `plugin_kernels/`

This directory holds the contracts of external kernel libraries that load as
SGLang plugins (the `sglang.srt.plugins` entry point) and replace a model's step
with kernels specialized for one model, configuration and platform: today the
small-batch decode and verify steps, possibly prefill or high-throughput steps
later. `KernelBackend.PLUGIN` records them. These are kernel plugins, not the
platform plugins that add a device. The kernels stay in the libraries, and the
code is split at the kernel/runtime boundary:

- `kernels/plugin_kernels/<hardware>/` holds what a plugin implements and
  registers: its spec (registration, API version, requirements, capability and
  width declarations, the build and forward contract) and any thin wrappers
  around its kernels, under `<hardware>/<model>/`. It imports nothing from
  `srt`.
- `srt/kernel_plugins/<hardware>/binding.py` holds SGLang's integration: it
  reads the server configuration at argument resolution, observes weight
  loading, coordinates TP ranks, plans CUDA graph coverage, installs the
  model-forward dispatch with its fallback, and hands a plugin the runtime
  state it may use.

| Hardware | Server flag | Library |
| --- | --- | --- |
| `nvidia/` | `--enable-nvidia-megakernel` | [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) |
| `amd/` | `--enable-amd-monokernel` | [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) |

An operator exposed from here registers into its logical group with
`KernelSpec(op="<group>.<name>", backend=KernelBackend.PLUGIN, target="sglang.kernels.plugin_kernels.<hardware>.<model>.<mod>:<fn>")`;
`test_kernels_namespace.py` enforces the pairing. See `plugin_kernels/README.md`.

## How it works

Each `ops.<group>` function is a thin wrapper that forwards to a chosen
backend, and every backend is described by a `KernelSpec` in the registry so
alternatives can be inventoried and compared:

- `register_kernel(KernelSpec(...))` records metadata only — an operator id
  (`"<group>.<name>"`), a backend, and an import path (`"module:attr"`). No
  `torch` or kernel backend is imported, and no JIT compilation is triggered,
  until a kernel is actually called.
- `select_kernel(op, backend=None)` resolves an op to its fixed call path.
  There is **no** priority ranking or heuristic auto-selection: an op with a
  single backend resolves to it; an op with several backends must be resolved
  by naming one (`backend=...`). The extra backends are inventory only.
- `get_kernel(op, backend)` resolves and caches the callable; the public
  wrappers use it, pinned to the backend whose signature they document.

The public wrappers currently default to the AOT `sgl_kernel` implementation
(the stable wheel boundary, broadest shape support). The JIT CUDA backend is
registered alongside for inventory; where its signature differs, select it
explicitly, e.g.:

```python
from sglang.kernels import select_kernel, KernelBackend
jit_moe_align = select_kernel("moe.moe_align_block_size", backend=KernelBackend.JIT).load()
```

## `BaseFusedOp` — the unified operator contract

`BaseFusedOp` is a standard `torch.nn.Module` (it replaced the former
`sglang.srt.layers.utils.MultiPlatformOp`) that carries one logical operator
with interchangeable implementations along **two independent dimensions**:

- **Kernel backends (provenance)** — one `forward_<backend>` method per
  implementation source, all sharing one signature behind a single
  `forward()`:
  - `forward_native` — **required**; the pure-`torch` correctness reference
    every other implementation is checked against.
  - `forward_torch_compile` — inherited for free as
    `torch.compile(forward_native)`.
  - `forward_triton` / `forward_jit` / `forward_aot` / `forward_cute_dsl` /
    `forward_helion` / `forward_kda` / `forward_cake` / `forward_plugin` /
    `forward_flashinfer` / `forward_deepgemm` / `forward_aiter` /
    `forward_torch_npu` — opt-in overrides. A backend is *available* iff its
    method is overridden; it joins **auto-selection** only when also declared
    in `capabilities` (device support is metadata, not guesswork).
- **Platforms / devices** — optional composite per-device paths:
  `forward_cuda`, `forward_hip` (falls back to `forward_cuda`),
  `forward_npu`, `forward_xpu`, `forward_musa` (no implicit CUDA fallback —
  MUSA ops opt into the CUDA path with an explicit `forward_musa`),
  `forward_cpu` (AMX CPUs), plus `forward_<key>` /
  `register_oot_forward()` for out-of-tree platform plugins. CUDA / HIP are
  **not** kernel backends.

Dispatch priority, highest first: explicit `forward(..., backend=...)` →
global forced backend (`SGLANG_FORCE_FUSED_OP_BACKEND`) → OOT platform
override → declared optimized kernel backends by `priority` (filtered by
`backend_eligible()`, a `CapabilityRequirement`-vs-`PlatformInfo` check
extensible with per-call applicability gates) → platform-specific forward →
`forward_native`. A gate only answers whether the kernel covers these operands,
so that falling through is a correct way to compute the same result; operands
that violate the op's contract (mismatched shapes or devices) are a caller bug
that the backend's launcher must raise, not a reason to fall back. The static part of the decision is resolved once and cached
on the instance, so the hot path stays a single indirect call.

`BaseFusedOp` also owns the torch.compile mode protocol
(`enter_torch_compile(num_tokens)` / `leave_torch_compile()`, both
idempotent): while an outer model is compiled, ops switch to their
compile-safe native path so device kernels are never traced (TopK and Fused
MoE override `_torch_compile_forward()` to keep their bs>1 behavior).

The public `ops.<group>` functions stay thin wrappers over
module-level instances, so the import surface is unchanged; each instance also
registers all of its backends as `KernelSpec`s so the registry inventory and
`select_kernel(..., backend=...)` keep working.

An op that is built per layer around the model's own modules (e.g. the Muse
Spark norms) has no module-level instance; its group registers each backend
method as `module:Class.forward_<backend>` for the inventory.

What this buys (see the
[RFC discussion](https://github.com/sgl-project/sglang/issues/29630#issuecomment-4920387930)):

- **Unified correctness testing** — a generic harness enumerates
  `available_backends()` and asserts each one matches `forward_native`
  (`test/registered/kernels/ops/activation/test_fused_op_gpu_parity.py`); new
  backends are picked up automatically.
- **One-switch debugging** — `SGLANG_FORCE_FUSED_OP_BACKEND=torch` (or
  `set_fused_op_backend(KernelBackend.TORCH)`) flips *every* fused op to its
  reference implementation for numerical-bug bisection.
- **Safe fallbacks** — a missing / ineligible optimized kernel degrades to
  `native` instead of scattering `if`/`else` at call sites.
- **Incremental optimization** — land `forward_native` first, add `triton` /
  `cuda_jit` / `cuda_aot` later without touching call sites; alternative
  implementations of the same op live side by side for A/B.
- **Tracing** — `enable_fused_op_trace()` records every call's op, backend,
  and tensor shapes/dtypes, giving an accurate inventory of what a model
  actually exercises.

## Review rule (RFC #29630)

> SGLang runtime code and tests should import callable kernels from
> `sglang.kernels.ops.*`.

Implementation work can still happen in `sglang.kernels.jit` or `sgl_kernel`.
When a PR adds a new callable kernel, add a `sglang.kernels.ops.*` entry point
for it, and avoid growing `sglang.kernels.jit` as a long-term public operator
namespace.
