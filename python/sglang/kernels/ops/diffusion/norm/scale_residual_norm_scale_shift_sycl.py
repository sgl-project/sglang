import torch
from sgl_kernel.diffusion import (
    can_use_fused_scale_residual_norm_scale_shift as can_use_fused_scale_residual_norm_scale_shift_sycl,
)
from sgl_kernel.diffusion import (
    fused_scale_residual_norm_scale_shift as fused_scale_residual_norm_scale_shift_sycl,
)

from sglang.srt.utils.patch_torch import register_fake_if_exists


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
