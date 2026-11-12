"""Triton RMSNorm (optionally fused with the residual add) for ROCm builds without
vLLM's or aiter's kernels, with RMSNorm.forward_native's semantics: FP32 math, the
residual sum rounded to the input dtype, and the weight applied in FP32 before the
final rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    R,
    W,
    Y,
    RO,
    stride_x,
    stride_r,
    stride_y,
    stride_ro,
    N: tl.constexpr,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        x = x + tl.load(R + row * stride_r + cols, mask, other=0).to(tl.float32)
        tl.store(RO + row * stride_ro + cols, x.to(RO.dtype.element_ty), mask)
    inv = tl.math.rsqrt(tl.sum(x * x, 0) / N + eps)
    w = tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (x * inv * w).to(Y.dtype.element_ty), mask)


def rmsnorm(x, weight, eps, residual=None):
    """RMSNorm over the last dim; with residual, (norm(x + residual), x + residual)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    r2 = ro = None
    if residual is not None:
        r2 = residual.reshape(-1, shape[-1])
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
        # forward_native returns the residual sum in x's dtype.
        ro = torch.empty_like(r2, dtype=x2.dtype)
    rows, n = x2.shape
    if rows:
        block = triton.next_power_of_2(n)
        _rmsnorm_kernel[(rows,)](
            x2,
            r2,
            weight,
            y,
            ro,
            x2.stride(0),
            r2.stride(0) if r2 is not None else 0,
            y.stride(0),
            ro.stride(0) if ro is not None else 0,
            n,
            eps,
            residual is not None,
            block,
            num_warps=min(16, max(1, block // 512)),
        )
    if residual is None:
        return y.view(shape)
    return y.view(shape), ro.view(residual.shape)
