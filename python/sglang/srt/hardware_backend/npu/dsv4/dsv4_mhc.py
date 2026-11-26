"""A3 DSV4.1 residual mixing after a sublayer."""

from __future__ import annotations

import torch


def post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Accumulate residual mixing and the sublayer output in FP32."""
    return (
        torch.bmm(comb.float().transpose(1, 2), residual.float())
        + post_mix.float().unsqueeze(-1) * x.float().unsqueeze(1)
    ).to(x.dtype)
