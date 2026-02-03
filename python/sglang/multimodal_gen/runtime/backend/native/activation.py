import math

import torch
import torch.nn.functional as F


def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def gelu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]


def gelu_new(self, x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


def quick_gelu(self, x: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation equivalent to forward()."""
    return x * torch.sigmoid(1.702 * x)
