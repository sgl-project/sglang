"""Import helpers for speculative kernel modules that may not launch Triton.

This keeps these modules importable without Triton, but does not make every
higher-level speculative-decoding dependency Triton-free.
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
