import torch
from torch import nn


def apply_group_norm_silu(
    x: torch.Tensor,
    norm: nn.Module,
    activation: nn.Module,
) -> torch.Tensor:
    if (
        x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and isinstance(norm, nn.GroupNorm)
        and isinstance(activation, nn.SiLU)
        and not activation.inplace
        and norm.affine
        and norm.weight is not None
        and norm.bias is not None
    ):
        from sglang.jit_kernel.diffusion.triton.group_norm_silu import (
            triton_group_norm_silu,
        )

        return triton_group_norm_silu(
            x,
            norm.weight,
            norm.bias,
            num_groups=norm.num_groups,
            eps=norm.eps,
        )
    return activation(norm(x))


__all__ = ["apply_group_norm_silu"]
