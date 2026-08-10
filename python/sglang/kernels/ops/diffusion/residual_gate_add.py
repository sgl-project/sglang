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

logger = logging.getLogger(__name__)


@cache_once
def _jit_residual_gate_add_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported residual_gate_add dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_residual_gate_add",
        *args,
        cuda_files=["diffusion/residual_gate_add.cuh"],
        cuda_wrappers=[
            (
                "residual_gate_add",
                "residual_gate_add::" f"ResidualGateAddKernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(residual)


@register_custom_op(
    op_name="diffusion_residual_gate_add",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _residual_gate_add_custom_op(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(residual)
    module = _jit_residual_gate_add_module(residual.dtype)
    broadcast_gate = gate.shape != residual.shape
    module.residual_gate_add(
        out.view(-1),
        residual.view(-1),
        update.view(-1),
        gate.view(-1),
        residual.shape[-1],
        broadcast_gate,
    )
    return out


def _is_row_broadcast_gate(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    if gate.dim() != residual.dim() or gate.shape[-1] != residual.shape[-1]:
        return False
    return all(size == 1 for size in gate.shape[:-1])


def can_use_residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    return (
        residual.dtype in _SUPPORTED_DTYPES
        and residual.dtype == update.dtype
        and residual.dtype == gate.dtype
        and residual.is_cuda
        and update.is_cuda
        and gate.is_cuda
        and residual.device == update.device == gate.device
        and residual.dim() >= 2
        and residual.numel() > 0
        and update.shape == residual.shape
        and (gate.shape == residual.shape or _is_row_broadcast_gate(residual, gate))
        and residual.is_contiguous()
        and update.is_contiguous()
        and gate.is_contiguous()
    )


def residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if not can_use_residual_gate_add_cuda(residual, update, gate):
        raise RuntimeError("unsupported input for residual_gate_add CUDA")
    return _residual_gate_add_custom_op(residual, update, gate)


def residual_gate_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Use the bit-exact CUDA fast path when supported, otherwise eager.

    Runtime build failures are cached per device and dtype so every diffusion
    model shares one fallback policy instead of maintaining model-local flags.
    """
    runtime_key = (residual.device.index, residual.dtype)
    if (
        residual.dtype in _BIT_EXACT_DTYPES
        and runtime_key not in _FAILED_RUNTIME_KEYS
        and can_use_residual_gate_add_cuda(residual, update, gate)
    ):
        try:
            return residual_gate_add_cuda(residual, update, gate)
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_KEYS.add(runtime_key)
            logger.warning(
                "Disabling diffusion residual-gate CUDA fast path on %s/%s: %s",
                residual.device,
                residual.dtype,
                exc,
            )
    return residual + update * gate


__all__ = [
    "can_use_residual_gate_add_cuda",
    "residual_gate_add",
    "residual_gate_add_cuda",
]
