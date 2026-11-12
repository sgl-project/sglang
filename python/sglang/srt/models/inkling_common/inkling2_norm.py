import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        pre_affine: bool = False,
        output_dtype: torch.dtype | None = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.output_dtype = output_dtype
        self.weight = (
            nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        )
        self.pre_weight = nn.Parameter(torch.ones(hidden_size)) if pre_affine else None

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        from sglang.kernels.ops.inkling2.norm import rms_norm

        return rms_norm(
            x,
            self.weight,
            self.eps,
            pre_weight=self.pre_weight,
            output_dtype=self.output_dtype,
            out=out,
        )
