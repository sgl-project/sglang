"""Adapters for the Ascend fusions published by sgl-kernel-npu.

The kernels and their public operators live in sgl-kernel-npu
(``sgl_kernel_npu.operators.diffusion``); this module only resolves them from
the installed wheel, once per process. A wheel that predates them leaves the
fusions unavailable so the caller keeps its eager chain, instead of failing a
model that would otherwise run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UNRESOLVED = object()
# (silu_mul, residual_gate_add, rmsnorm_cast_before_weight) or None.
_ops: object = _UNRESOLVED


def _resolve():
    global _ops
    if _ops is not _UNRESOLVED:
        return _ops
    try:
        from sgl_kernel_npu.operators.diffusion import (
            residual_gate_add,
            rmsnorm_cast_before_weight,
            silu_mul,
        )
    except ImportError:
        logger.warning(
            "sgl-kernel-npu does not provide sgl_kernel_npu.operators.diffusion; "
            "Qwen-Image 2.1 keeps its eager NPU path. Install a kernel wheel "
            "from the matching sgl-kernel-npu release."
        )
        _ops = None
    else:
        _ops = (silu_mul, residual_gate_add, rmsnorm_cast_before_weight)
    return _ops


def _op(index: int):
    ops = _resolve()
    if ops is None:
        raise RuntimeError(
            "Qwen-Image 2.1 NPU fusions need sgl_kernel_npu.operators.diffusion"
        )
    return ops[index]


def npu_fusions_available() -> bool:
    """Whether the installed sgl-kernel-npu wheel carries the three operators."""
    return _resolve() is not None


def silu_mul_npu(gate, value):
    return _op(0)(gate, value)


def residual_gate_add_npu(residual, update, gate):
    return _op(1)(residual, update, gate)


def rmsnorm_npu_onepass(x, weight, eps):
    return _op(2)(x, weight, eps)
