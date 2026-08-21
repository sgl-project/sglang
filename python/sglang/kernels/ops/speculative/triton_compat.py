"""Triton imports that tolerate a missing triton install.

Kernel modules under this package import triton at module scope. Builds
without triton (e.g. some Ascend NPU images) must still be able to import
them so device-dispatched torch fallbacks can load; only *launching* a
Triton kernel is an error there, and those code paths never run.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    jit = triton.jit
    has_triton = True
except ImportError:  # pragma: no cover - needs a triton-free environment
    triton = None
    has_triton = False

    class _MissingTritonLanguage:
        constexpr = object

    class _MissingTritonKernel:
        """Import-time stand-in for a @triton.jit function; launch raises."""

        def __init__(self, name: str) -> None:
            self._name = name

        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                raise RuntimeError(
                    f"{self._name} is a Triton kernel but triton is not "
                    "installed in this environment; this code path only "
                    "runs on CUDA devices."
                )

            return _launch

    def jit(fn):
        return _MissingTritonKernel(fn.__name__)

    tl = _MissingTritonLanguage()


def next_power_of_2(n: int) -> int:
    if has_triton:
        return triton.next_power_of_2(n)
    return 1 << (int(n) - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    if has_triton:
        return triton.cdiv(a, b)
    return -(-int(a) // int(b))
