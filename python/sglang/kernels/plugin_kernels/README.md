# `plugin_kernels` — plugin kernel backend

The contracts of external kernel libraries that load as SGLang plugins and
replace a model's step with kernels specialized for one model, configuration
and platform. Today they serve small-batch decode and verify steps; prefill or
high-throughput specializations fit here too. `KernelBackend.PLUGIN` records
them; the [kernels README](../README.md) explains how a backend relates to the
operator groups. These are kernel plugins, not the platform plugins that add a
device.

## Layout

The code is split at the kernel/runtime boundary.

```
kernels/plugin_kernels/      # provider contracts and registration
  megakernel/spec.py         #   shared megakernel contract and target-aware registry
  nvidia/spec.py             #   NVIDIA settings and compatibility imports
  amd/spec.py                #   AMD monokernel contract and registry
  <hardware>/<model>/        #   thin wrappers around the library's kernels
srt/kernel_plugins/          # SGLang's integration
  <hardware>/binding.py      #   startup checks, weight observation, TP agreement,
                             #   CUDA graph coverage, forward dispatch and fallback,
                             #   and the runtime state handed to the plugin
```

| Hardware | Server flag | Library |
| --- | --- | --- |
| `nvidia/` | `--enable-nvidia-megakernel` | [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) |
| `amd/` | `--enable-amd-monokernel` | [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) |

## How a plugin runs

A library registers its specs through the `sglang.srt.plugins` entry point.
With its server flag set, `srt/kernel_plugins/<hardware>/binding.py`:
- checks the spec against the resolved configuration at argument resolution,
  before the weights load, and stops the server listing every reason it cannot
  serve;
- builds the spec once the model is prepared;
- replaces that model's forward for the steps the spec covers; every other step
  runs the model's own forward.

The kernels stay in their libraries. Megakernel providers import the shared
contract from `sglang.kernels.plugin_kernels.megakernel.spec`. Existing NVIDIA
providers can keep importing `sglang.kernels.plugin_kernels.nvidia.spec`. AMD
monokernel providers import `sglang.kernels.plugin_kernels.amd.spec`. The
runtime integrations live in `sglang.srt.kernel_plugins.<hardware>`. Unit tests are in
`test/registered/unit/kernel_plugins/<hardware>/`.

## Operators

An external kernel exposed as an operator registers into its logical group
(`ops/<group>/__init__.py`) with `KernelSpec(op="<group>.<name>",
backend=KernelBackend.PLUGIN,
target="sglang.kernels.plugin_kernels.<hardware>.<model>.<module>:<function>")`;
callers import the `ops.<group>` facade.
