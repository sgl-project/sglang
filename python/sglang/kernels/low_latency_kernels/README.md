# `low_latency_kernels` — low-latency kernel backend

The integration of external low-latency kernel libraries, whose kernels run
small-batch decode where launch gaps and collective latency dominate a step.
`KernelBackend.LOW_LATENCY` records them; the [kernels README](../README.md)
explains how a backend relates to the operator groups.

## Layout

```
low_latency_kernels/
  <hardware>/    # the whole-step interface to that hardware's library
    <model>/     # model-specific integration code
```

| Hardware | Server flag | Library |
| --- | --- | --- |
| `nvidia/` | `--enable-nvidia-megakernel` | [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) |
| `amd/` | `--enable-amd-monokernel` | [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) |

## Whole-step interfaces

Each interface replaces a covered model's forward for the decode and verify
steps its library serves; every other step runs the model's own forward. It
holds the spec contract the library implements (`spec.py`), the startup checks
run at argument resolution, before the weights load (`guards.py`), and the
build and per-step routing installed on the model (`runtime.py`;
`nvidia/binding.py` is the runtime state handed to a megakernel). An
unsupported setup stops the server at startup and lists every reason.

The kernels stay in their libraries, which register their specs through the
`sglang.srt.plugins` entry point. The library versions that `docker/Dockerfile`
and `python/pyproject_other.toml` pin import these modules as
`sglang.srt.megakernels` and `sglang.srt.monokernels`, which alias them. Unit
tests are in `test/registered/unit/kernels/low_latency_kernels/<hardware>/`.

## Operators

An external low-latency kernel exposed as an operator registers into its
logical group (`ops/<group>/__init__.py`) with
`KernelSpec(op="<group>.<name>", backend=KernelBackend.LOW_LATENCY,
target="sglang.kernels.low_latency_kernels.<hardware>.<model>.<module>:<function>")`;
callers import the `ops.<group>` facade.
