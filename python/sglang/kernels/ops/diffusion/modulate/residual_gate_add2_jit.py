from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BIT_EXACT_DTYPES = (torch.float16, torch.bfloat16)
_FAILED_RUNTIME_KEYS: set[tuple[int | None, torch.dtype]] = set()
# Below this the pass is launch-bound and the custom-op dispatch costs more
# than the saved pass: on GB10 at 320x3072 bf16 the fused call is 15 us vs
# 13 us eager, while at 27280x3072 it is 2.92 ms vs 4.52 ms.
_MIN_FUSED_NUMEL = 4 << 20

logger = logging.getLogger(__name__)


@cache_once
def _jit_residual_gate_add2_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported residual_gate_add2 dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_residual_gate_add2",
        *args,
        cuda_files=["diffusion/residual_gate_add2.cuh"],
        cuda_wrappers=[
            (
                "residual_gate_add2",
                f"residual_gate_add2::ResidualGateAdd2Kernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


@register_custom_op(
    op_name="diffusion_residual_gate_add2",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _residual_gate_add2_custom_op(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    module = _jit_residual_gate_add2_module(residual.dtype)
    broadcast_gate = gate.shape != residual.shape
    module.residual_gate_add2(
        out.view(-1),
        residual.view(-1),
        update_a.view(-1),
        update_b.view(-1),
        gate.view(-1),
        residual.shape[-1],
        broadcast_gate,
    )
    return out


def _is_row_broadcast_gate(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    if gate.dim() != residual.dim() or gate.shape[-1] != residual.shape[-1]:
        return False
    return all(size == 1 for size in gate.shape[:-1])


def can_use_residual_gate_add2_cuda(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> bool:
    tensors = (residual, update_a, update_b, gate)
    return (
        residual.numel() >= _MIN_FUSED_NUMEL
        and all(t.dtype == residual.dtype for t in tensors)
        and residual.dtype in _SUPPORTED_DTYPES
        and all(t.is_cuda for t in tensors)
        and all(t.device == residual.device for t in tensors)
        and residual.dim() >= 2
        and update_a.shape == residual.shape
        and update_b.shape == residual.shape
        and (gate.shape == residual.shape or _is_row_broadcast_gate(residual, gate))
        and all(t.is_contiguous() for t in tensors)
    )


def residual_gate_add2_cuda(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if not can_use_residual_gate_add2_cuda(residual, update_a, update_b, gate):
        raise RuntimeError("unsupported input for residual_gate_add2 CUDA")
    return _residual_gate_add2_custom_op(residual, update_a, update_b, gate)


def residual_gate_add2(
    residual: torch.Tensor,
    update_a: torch.Tensor,
    update_b: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """``residual + gate * (update_a + update_b)`` in one pass.

    Bit-identical to ``torch.addcmul(residual, gate, update_a + update_b)``: the
    branch sum rounds to the tensor dtype and the gated product stays in fp32.
    Falls back to eager on unsupported inputs or a JIT build failure, cached per
    device and dtype like the single-update kernel.
    """
    runtime_key = (residual.device.index, residual.dtype)
    if (
        residual.dtype in _BIT_EXACT_DTYPES
        and runtime_key not in _FAILED_RUNTIME_KEYS
        and can_use_residual_gate_add2_cuda(residual, update_a, update_b, gate)
    ):
        try:
            return residual_gate_add2_cuda(residual, update_a, update_b, gate)
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_KEYS.add(runtime_key)
            logger.warning(
                "Disabling diffusion residual-gate2 CUDA fast path on %s/%s: %s",
                residual.device,
                residual.dtype,
                exc,
            )
    return torch.addcmul(residual, gate, update_a + update_b)


__all__ = [
    "can_use_residual_gate_add2_cuda",
    "residual_gate_add2",
    "residual_gate_add2_cuda",
]
