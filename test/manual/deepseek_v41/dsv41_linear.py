import torch
from torch import nn

from sglang.srt.layers.attention.dsv4.torch_quant import (
    FP4_BLOCK_SIZE,
    FP8_BLOCK_SIZE,
    naive_linear,
)


class Linear(nn.Module):
    """Weight kept in its checkpoint dtype: fp8 / fp4 carry a ue8m0 `scale`, bf16 / fp32 do not."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if dtype == torch.float4_e2m1fn_x2:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features // 2, dtype=dtype)
            )
            self.scale = nn.Parameter(
                torch.empty(
                    out_features,
                    in_features // FP4_BLOCK_SIZE,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            self.scale = nn.Parameter(
                torch.empty(
                    (out_features + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
                    (in_features + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
                    dtype=torch.float8_e8m0fnu,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype)
            )
            self.register_parameter("scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return naive_linear(x, self.weight, self.scale)
