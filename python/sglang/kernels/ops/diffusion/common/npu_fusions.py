"""Adapters for the Ascend fusions published by sgl-kernel-npu.

The kernels and their public operators live in sgl-kernel-npu
(``sgl_kernel_npu.operators.diffusion``); this module resolves them from the
installed wheel, once per process. A wheel that predates them raises
``ModuleNotFoundError`` at the first call, as every other NPU kernel path does.
"""

from __future__ import annotations

_UNRESOLVED = object()
# (silu_mul, rmsnorm_cast_before_weight)
_ops: object = _UNRESOLVED


def _resolve():
    global _ops
    if _ops is _UNRESOLVED:
        from sgl_kernel_npu.operators.diffusion import (
            rmsnorm_cast_before_weight,
            silu_mul,
        )

        _ops = (silu_mul, rmsnorm_cast_before_weight)
    return _ops


def silu_mul_npu(gate, value):
    return _resolve()[0](gate, value)


def rmsnorm_npu_onepass(x, weight, eps):
    return _resolve()[1](x, weight, eps)
