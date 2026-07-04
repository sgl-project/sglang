# `sglang.kernels` — unified kernel namespace

This package is the public in-tree import surface for callable kernels, per
[RFC #29630](https://github.com/sgl-project/sglang/issues/29630).

```python
from sglang.kernels.ops.layernorm import rmsnorm
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.kvcache import reshape_and_cache_flash
```

## Layout

```
sglang/kernels/
  spec.py        # KernelSpec, KernelBackend, FormatSignature,
                 # CapabilityRequirement, PlatformInfo
  registry.py    # process-wide KernelRegistry + register_kernel()
  selector.py    # heuristic select_kernel() and cached get_kernel()
  ops/
    <group>/     # one subpackage per operator group
```

Groups populated in this phase: `activation`, `gemm`, `kvcache`, `layernorm`,
`moe`, `quantization`. The remaining groups (`attention`, `communication`,
`diffusion`, `grammar`, `mamba`, `memory`, `sampling`, `spatial`,
`speculative`) are reserved package placeholders whose implementations still
live in `sglang.jit_kernel` / `sgl_kernel` / `triton_ops` and will migrate in
later phases.

## How it works

Implementations are not moved yet. Each `ops.<group>` function is a thin
wrapper that forwards to a chosen backend, and every backend is described by a
`KernelSpec` in the registry so alternatives can be inventoried and compared:

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
jit_rmsnorm = select_kernel("layernorm.rmsnorm", backend=KernelBackend.CUDA_JIT).load()
```

## Review rule (RFC #29630)

> SGLang runtime code and tests should import callable kernels from
> `sglang.kernels.ops.*`.

Implementation work can still happen in `sglang.jit_kernel` or `sgl_kernel`.
When a PR adds a new callable kernel, add a `sglang.kernels.ops.*` entry point
for it, and avoid growing `sglang.jit_kernel` as a long-term public operator
namespace.
