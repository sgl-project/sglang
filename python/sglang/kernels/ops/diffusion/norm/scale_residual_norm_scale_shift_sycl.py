import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.diffusion.norm.scale_residual_norm_scale_shift_triton import (
    can_use_fused_scale_residual_norm_scale_shift_triton,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

_fused_scale_residual_norm_scale_shift = (
    torch.ops.sgl_kernel.fused_scale_residual_norm_scale_shift
)


def _param_dtype(
    *, gate: torch.Tensor | int, weight: torch.Tensor | None, shift: torch.Tensor
) -> torch.dtype:
    if weight is not None:
        return weight.dtype
    if isinstance(gate, torch.Tensor):
        return gate.dtype
    return shift.dtype


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
    if not can_use_fused_scale_residual_norm_scale_shift_triton(
        residual=residual,
        x=x,
        gate=gate,
        shift=shift,
        scale=scale,
        weight=weight,
        bias=bias,
    ):
        return False
    dtypes = {p.dtype for p in (gate, weight, bias) if isinstance(p, torch.Tensor)}
    if len(dtypes) > 1:
        return False
    param_dtype = _param_dtype(gate=gate, weight=weight, shift=shift)
    return all(
        m.dtype == param_dtype or param_dtype == torch.float32 for m in (shift, scale)
    )


def fused_scale_residual_norm_scale_shift_sycl(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    param_dtype = _param_dtype(gate=gate, weight=weight, shift=shift)
    return _fused_scale_residual_norm_scale_shift(
        residual,
        x,
        gate if isinstance(gate, torch.Tensor) else None,
        weight,
        bias,
        scale.to(param_dtype),
        shift.to(param_dtype),
        eps,
    )


@register_fake_if_exists("sgl_kernel::fused_scale_residual_norm_scale_shift")
def _(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | None,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)
