import torch
from torch import nn

from sglang.srt.models.inkling_common.inkling2_arch import prefer_triton_fallbacks


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
        # The CUTE rms_norm compiles for Hopper and newer, so SM80 (A100) takes
        # the Triton path; every other arch keeps the existing kernel.
        self._use_triton = prefer_triton_fallbacks()

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if self._use_triton:
            from sglang.kernels.ops.layernorm.inkling2_rms_norm_triton import (
                rms_norm_fused as rms_norm,
            )
        else:
            from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm

        return rms_norm(
            x,
            self.weight,
            self.eps,
            pre_weight=self.pre_weight,
            output_dtype=self.output_dtype,
            out=out,
        )
