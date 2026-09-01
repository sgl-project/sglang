# Kernel Design Agent kernels

This directory contains optimized kernels produced by the Humanize2 workflow
([PolyArch/humanize](https://github.com/PolyArch/humanize)) together with
[Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents).

Kernels in this tree are registered with the uppercase
`KernelBackend.KDA` provenance backend. Their implementation language may
still be CUDA, Triton, or CuTe DSL; `KDA` records how the candidate was
produced and qualified rather than which compiler built it.

Each kernel package must document its source task, exact source revision,
target hardware, supported shapes, and validation evidence. Serving dispatch
uses the capability and shape allowlist for kernels already qualified on
their target GPU.
