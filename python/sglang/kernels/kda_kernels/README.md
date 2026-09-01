# Kernel Design Agent kernels

This directory contains optimized kernels produced by the Humanize2 workflow
([PolyArch/humanize](https://github.com/PolyArch/humanize)) together with
[Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents).

Kernels in this tree are registered with the uppercase
`KernelBackend.KDA` provenance backend. Their implementation language may
still be CUDA, Triton, or CuTe DSL; `KDA` records how the candidate was
produced and qualified rather than which compiler built it.

Each kernel package must document its source task, exact source revision,
target hardware, supported shapes, and validation evidence. Generated kernels
remain opt-in until correctness and end-to-end serving performance have been
validated on their target GPU.

Earlier merged KDA diffusion kernels remain beside the diffusion operator
surface under ``sglang.kernels.ops.diffusion``. They use the same uppercase
``KernelBackend.KDA`` registration without moving mature implementation files
out of their operator domain; the backend is provenance, not a directory rule.
