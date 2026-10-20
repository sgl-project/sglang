"""Low-latency kernel backend (``KernelBackend.LOW_LATENCY``).

Small-batch decode kernels live here and register into their operator group with
this backend. ``megakernels`` and ``monokernels`` hold SGLang's whole-step
interfaces to the out-of-tree NVIDIA and AMD kernel libraries.
"""
