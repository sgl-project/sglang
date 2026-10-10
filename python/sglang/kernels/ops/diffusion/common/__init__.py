"""Shared infrastructure for the diffusion kernels -- no kernels of its own.

- ``platform``  : device predicates and the Triton-vs-fallback selector
- ``fallback_*``: pure-torch / NPU / MPS implementations for Triton-less devices
"""
