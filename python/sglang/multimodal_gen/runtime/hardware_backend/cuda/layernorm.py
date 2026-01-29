from typing import Optional, Tuple, Union

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm

from sglang.multimodal_gen.runtime.layers.triton_ops import (
    rms_norm_fn,
    triton_one_pass_rms_norm,
)


def rms_norm(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if residual is not None:
        residual_shape = residual.shape
        residual = residual.view(-1, shape[-1])

    if x.dtype == torch.float:
        # fp32
        out = rms_norm_fn(
            x, self.weight, bias=None, residual=residual, eps=self.variance_epsilon
        )
    elif residual is not None:
        fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
        return x.view(shape), residual.view(residual_shape)
    else:
        if x.shape[-1] <= 128:
            out = triton_one_pass_rms_norm(x, self.weight.data, self.variance_epsilon)
        else:
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    out = out.view(shape)
    return out


def layer_norm(
    self,
    x: torch.Tensor,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    shape = x.shape
    x = x.view(-1, self.hidden_size)
    return (
        self.norm_infer(
            x.view(-1, self.hidden_size),
            self.weight,
            self.bias,
            eps=self.eps,
            is_rms_norm=False,
        )
        .view(x.shape)
        .view(shape)
    )
