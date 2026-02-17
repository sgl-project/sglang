import torch
from sgl_kernel import silu_and_mul as silu_and_mul_kernel


def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    silu_and_mul_kernel(x, out)
    return out
