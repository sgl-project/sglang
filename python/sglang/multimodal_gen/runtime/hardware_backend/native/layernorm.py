from typing import Optional, Tuple, Union

import torch


def rms_norm(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    hidden_size = x.shape[-1]
    if hidden_size != self.hidden_size:
        raise ValueError(
            "Expected hidden_size to be "
            f"{self.hidden_size}, but found: {hidden_size}"
        )

    if self.variance_size_override is None:
        x_var = x
    else:
        if hidden_size < self.variance_size_override:
            raise ValueError(
                "Expected hidden_size to be at least "
                f"{self.variance_size_override}, but found: {hidden_size}"
            )

        x_var = x[..., : self.variance_size_override]

    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    x = (x * self.weight).to(orig_dtype)
    if residual is None:
        return x
    else:
        return x, residual


@torch.compile(backend="inductor")
def layer_norm(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    input_dtype = x.dtype
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + self.eps)
    if self.weight is not None:
        x = self.weight * x
    # if no affine, this is a no-op
    if self.bias is not None:
        x = x + self.bias
    return x.to(input_dtype)
