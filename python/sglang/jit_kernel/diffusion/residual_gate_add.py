from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@cache_once
def _jit_residual_gate_add_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_residual_gate_add",
        *args,
        cuda_files=["diffusion/residual_gate_add.cuh"],
        cuda_wrappers=[
            (
                "residual_gate_add",
                "sglang_residual_gate_add::" f"ResidualGateAddKernel<{args}>::run",
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
    module.residual_gate_add(out, residual, update, gate)
    return out


def _is_row_broadcast_gate(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    if gate.dim() != residual.dim() or gate.shape[-1] != residual.shape[-1]:
        return False
    row_dim = gate.dim() - 2
    return gate.shape[row_dim] == 1 and all(size == 1 for size in gate.shape[:-1])


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
