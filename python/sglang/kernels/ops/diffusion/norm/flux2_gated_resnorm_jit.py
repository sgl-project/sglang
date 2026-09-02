from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_HIDDEN = 6144
_ALIGNMENT = 32


def _blackwell_or_newer(device: torch.device) -> bool:
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 10
    )


def _aligned(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() % _ALIGNMENT == 0


def _row_bf16(tensor: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    if not (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype == torch.bfloat16
        and tensor.is_cuda
        and tensor.device == device
        and tensor.stride(-1) == 1
    ):
        return None
    if tensor.shape == (_HIDDEN,):
        row = tensor
    elif tensor.shape in ((1, _HIDDEN), (1, 1, _HIDDEN)):
        row = tensor.reshape(_HIDDEN)
    else:
        return None
    return row if _aligned(row) else None


def can_defer_flux2_gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> bool:
    if not (
        not torch.compiler.is_compiling()
        and residual.dtype == torch.bfloat16
        and residual.is_cuda
        and residual.dim() == 3
        and residual.shape[0] == 1
        and residual.shape[-1] == _HIDDEN
        and residual.numel() > 0
        and residual.is_contiguous()
        and _aligned(residual)
        and update.dtype == residual.dtype
        and update.device == residual.device
        and update.shape == residual.shape
        and update.is_contiguous()
        and _aligned(update)
        and _blackwell_or_newer(residual.device)
    ):
        return False
    return _row_bf16(gate, residual.device) is not None


def can_use_flux2_gated_resnorm(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return can_defer_flux2_gated_residual(residual, update, gate) and all(
        _row_bf16(tensor, residual.device) is not None for tensor in (scale, shift)
    )


@cache_once
def _module() -> Module:
    return load_jit(
        "flux2_gated_resnorm",
        cuda_files=["diffusion/flux2_gated_resnorm.cuh"],
        cuda_wrappers=[("run", "flux2_gated_resnorm::Kernel::run")],
    )


def flux2_gated_resnorm_raw(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    _module().run(
        output.view(-1, _HIDDEN),
        residual_out.view(-1, _HIDDEN),
        residual.view(-1, _HIDDEN),
        update.view(-1, _HIDDEN),
        _row_bf16(gate, residual.device),
        _row_bf16(scale, residual.device),
        _row_bf16(shift, residual.device),
        float(eps),
    )
    return output, residual_out


__all__ = [
    "can_defer_flux2_gated_residual",
    "can_use_flux2_gated_resnorm",
    "flux2_gated_resnorm_raw",
]
