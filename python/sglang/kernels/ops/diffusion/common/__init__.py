"""Shared infrastructure for the diffusion kernels -- no kernels of its own.

- ``platform``  : device predicates and the Triton-vs-fallback selector
- ``fallback_*``: pure-torch / NPU / MPS implementations for Triton-less devices
- ``fast_launch``: signature-cached launches for small Triton kernels
- ``jit_arch``  : the SM80 floor the CUDA JIT builders refuse below
"""
