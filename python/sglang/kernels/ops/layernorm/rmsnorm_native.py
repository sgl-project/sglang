"""RMSNorm bit-identical to ``RMSNorm.forward_native`` on CUDA, in 3 launches.

With ``residual``, it also folds in the ``x + residual`` add that precedes the
norm in a pre-norm block (rounded to the input dtype, as the eager add is).

``forward_native`` runs eight kernels (cast, square, mean, add, rsqrt, two
multiplies, cast) and moves the FP32 copies through memory between them. Here
the squares come from one kernel, the row means from the same ``torch.mean``
reduction (so the summation order, hence every bit of the variance, matches),
and the normalization from a second kernel that keeps the reference's
operation order and rounding.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import cuda_rsqrtf

_BLOCK = 1024


@triton.jit
def _square_kernel(
    X,
    Residual,
    Sum,
    Squared,
    n_elements,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(X + idx, mask, other=0)
    if HAS_RESIDUAL:
        r = tl.load(Residual + idx, mask, other=0)
        # Same as the eager ``x + residual``: fp32 add, rounded to the dtype.
        x = (x.to(tl.float32) + r.to(tl.float32)).to(Sum.dtype.element_ty)
        tl.store(Sum + idx, x, mask)
    x = x.to(tl.float32)
    tl.store(Squared + idx, x * x, mask)


@triton.jit
def _normalize_kernel(
    X,
    Variance,
    Weight,
    Out,
    n_elements,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(X + idx, mask, other=0).to(tl.float32)
    variance = tl.load(Variance + idx // DIM, mask, other=1.0)
    weight = tl.load(Weight + idx % DIM, mask, other=0)
    # (x * rsqrt(variance + eps)) * weight, as forward_native orders it.
    normalized = x * cuda_rsqrtf(variance + EPS)
    tl.store(Out + idx, normalized * weight, mask)


def rmsnorm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
):
    """``RMSNorm.forward_native(x)`` for an FP32 ``weight``, without its own
    residual handling. With ``residual``, normalizes ``s = x + residual`` and
    returns ``(norm(s), s)``."""
    assert x.is_cuda and weight.dtype == torch.float32 and weight.is_contiguous()
    assert weight.shape == (x.shape[-1],) and weight.device == x.device
    x = x.contiguous()
    total = None
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype
        residual = residual.contiguous()
        total = torch.empty_like(x)
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out if residual is None else (out, total)
    squared = torch.empty_like(x, dtype=torch.float32)
    grid = (triton.cdiv(n_elements, _BLOCK),)
    _square_kernel[grid](
        x,
        residual if residual is not None else x,
        total if total is not None else x,
        squared,
        n_elements,
        HAS_RESIDUAL=residual is not None,
        BLOCK=_BLOCK,
        enable_fp_fusion=False,
    )
    normed = total if total is not None else x
    variance = squared.mean(dim=-1, keepdim=True)
    _normalize_kernel[grid](
        normed,
        variance,
        weight,
        out,
        n_elements,
        DIM=x.shape[-1],
        EPS=eps,
        BLOCK=_BLOCK,
        enable_fp_fusion=False,
    )
    return out if residual is None else (out, total)
