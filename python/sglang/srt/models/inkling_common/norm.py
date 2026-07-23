from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from sgl_kernel import rmsnorm
except ImportError:
    rmsnorm = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if not x.is_cuda or rmsnorm is None:
            return F.rms_norm(
                x, (self.hidden_size,), self.weight, self.variance_epsilon
            )

        original_shape = x.shape
        if original_shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"RMSNorm expected hidden size {self.hidden_size}, got {original_shape[-1]}"
            )
        x_2d = x.reshape(-1, self.hidden_size)
        try:
            y = rmsnorm(x_2d, self.weight.to(x_2d.dtype), self.variance_epsilon)
        except (AttributeError, RuntimeError):
            return F.rms_norm(
                x, (self.hidden_size,), self.weight, self.variance_epsilon
            )
        return y.view(original_shape)
