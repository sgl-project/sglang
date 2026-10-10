import torch
from sgl_kernel.diffusion import (
    MAX_FUSED_HIDDEN,
)
from sgl_kernel.diffusion import (  # noqa: F401  re-exported via ops.diffusion
    fused_scale_residual_norm_scale_shift as fused_scale_residual_norm_scale_shift_sycl,
)

from sglang.srt.utils.patch_torch import register_fake_if_exists

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def can_use_fused_scale_residual_norm_scale_shift_sycl(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> bool:
    # sgl_kernel's wrapper raises on unsupported operands; this lets the caller
    # fall back to triton or eager instead.
    if x.device.type != "xpu" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.dim() != 3 or x.shape[0] != 1 or not x.is_contiguous():
        return False
    for operand in (residual, gate, shift, scale, weight, bias):
        if isinstance(operand, torch.Tensor) and (
            operand.device.type != "xpu" or not operand.is_contiguous()
        ):
            return False
    if residual.shape != x.shape or residual.dtype != x.dtype:
        return False
    hidden = x.shape[-1]
    if hidden > MAX_FUSED_HIDDEN:
        return False
    if isinstance(gate, torch.Tensor):
        if gate.dim() not in (3, 4) or gate.shape[0] != 1 or gate.shape[-1] != hidden:
            return False
        if gate.dim() == 3:
            if gate.shape[1] != 1:
                return False
        elif (
            gate.shape[2] != 1 or gate.shape[1] <= 0 or x.shape[1] % gate.shape[1] != 0
        ):
            return False
    elif type(gate) is not int or gate != 1:
        return False
    for modulation in (scale, shift):
        if not isinstance(modulation, torch.Tensor):
            return False
        if modulation.numel() not in (1, hidden):
            return False
    for affine in (weight, bias):
        if affine is not None and affine.numel() != hidden:
            return False
    params = (gate, shift, scale, weight, bias)
    dtypes = {p.dtype for p in params if isinstance(p, torch.Tensor)}
    return len(dtypes) == 1 and shift.dtype in _SUPPORTED_DTYPES


@register_fake_if_exists("sgl_kernel::fused_scale_residual_norm_scale_shift")
def _fused_scale_residual_norm_scale_shift_sycl_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | None,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = x.new_empty(x.shape)
    residual_out = x.new_empty(x.shape)
    return y, residual_out
