"""Out-of-place RMSNorm with FP32 statistics and weight multiplication."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fp32_kernel(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    weight = tl.load(W + col, col < D, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    normalized = x * inv_rms
    tl.store(Y + row * D + col, normalized * weight, col < D)


def rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match ``weight * (x.float() * rsqrt(mean(x.float()**2) + eps))``."""
    assert x.is_cuda and x.is_contiguous() and weight.is_contiguous()
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert weight.dtype in (torch.bfloat16, torch.float32)
    assert x.device == weight.device and weight.shape == (x.shape[-1],)
    output = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    if rows:
        _rmsnorm_fp32_kernel[(rows,)](
            x,
            weight,
            output,
            dim,
            eps,
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
