# `low_latency_kernels` — low-latency kernel backend

Kernels for small-batch decode, where launch gaps and collective latency
dominate a step. `KernelBackend.LOW_LATENCY` records them; the
[kernels README](../README.md) explains how a backend relates to the operator
groups.

## Operator kernels

A low-latency operator kernel, such as one model's fused small-batch decode
chain, lives in a module or subpackage here named after what it serves, with
the JIT CUDA sources only it uses under `csrc/`. Shared JIT infrastructure
(`jit/include`, `jit/utils`) stays in `jit/`.

- Register each public op into its logical group (`ops/<group>/__init__.py`)
  with `KernelSpec(op="<group>.<name>", backend=KernelBackend.LOW_LATENCY,
  target="sglang.kernels.low_latency_kernels.<module>:<function>")`, classified
  by the same rules as any op; callers import the `ops.<group>` facade.
- The caller selects the kernel only for the shapes and configurations it
  serves and keeps the standard path for the rest.
- Numerical tests go in `test/registered/kernels/ops/<group>/`.

## Whole-step interfaces

| Server flag | Interface | Kernels |
| --- | --- | --- |
| `--enable-nvidia-megakernel` | `megakernels/` | [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) |
| `--enable-amd-monokernel` | `monokernels/` | [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) |

Each interface replaces a covered model's forward for the decode and verify
steps its plugin serves; every other step runs the model's own forward. It
holds the spec contract a plugin implements (`spec.py`), the startup checks
run at argument resolution, before the weights load (`guards.py`), and the
build and per-step routing installed on the model (`runtime.py`;
`megakernels/binding.py` is the runtime state handed to a megakernel). An
unsupported setup stops the server at startup and lists every reason.

The kernels stay in their own libraries, which register their specs through
the `sglang.srt.plugins` entry point. The plugin versions that
`docker/Dockerfile` and `python/pyproject_other.toml` pin import these modules
as `sglang.srt.megakernels` and `sglang.srt.monokernels`, which alias them.
Unit tests are in `test/registered/unit/kernels/low_latency_kernels/`.
