# Kernel Design Agent kernels

This directory contains optimized kernels produced through agentic kernel-design
workflows, including [Humanize2](https://github.com/PolyArch/humanize) and
[KDA-1.5](https://github.com/radixark/KDA-1.5).

Each kernel package must document its source task, exact source revision,
target hardware, supported shapes, and validation evidence. Generated kernels
remain opt-in until correctness and end-to-end serving performance have been
validated on their target GPU.
