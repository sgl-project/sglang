"""RMSNorm tail for a norm whose weight is folded into the next GEMM.

``RMSNorm(x; g) @ W^T == rstd(x) * (x @ (W * g)^T)`` row by row, so with
``a = x @ (W * g)^T`` the norm launch is replaced by scaling ``a`` here, fused
with the residual adds that follow the GEMM.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["stride_a", "stride_x", "stride_b", "stride_c", "stride_out"]
)
def _rstd_scale_add3_kernel(
    a_ptr,
    stride_a,
    x_ptr,
    stride_x,
    b_ptr,
    stride_b,
    c_ptr,
    stride_c,
    out_ptr,
    stride_out,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    # Loading the [N] slices before the row reduction and a runtime output stride
    # took 300 rows x 7168 from 7.4 to 5.1 us on MI355X.
    a = tl.load(a_ptr + row * stride_a + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * stride_b + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if HAS_C:
        c = tl.load(c_ptr + row * stride_c + offs_n, mask=mask_n, other=0.0)
        c = c.to(tl.float32)
    offs_k = tl.arange(0, K_POW2)
    x = tl.load(x_ptr + row * stride_x + offs_k, mask=offs_k < K, other=0.0)
    x = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)

    out_ty = out_ptr.dtype.element_ty
    # Same rounding points as the unfused norm -> GEMM -> add -> add chain.
    y = (a * rstd).to(out_ty)
    y = (y.to(tl.float32) + b).to(out_ty)
    if HAS_C:
        y = (y.to(tl.float32) + c).to(out_ty)
    tl.store(out_ptr + row * stride_out + offs_n, y, mask=mask_n)


def rstd_scale_add3(
    a: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    eps: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out = bf16(bf16(bf16(a * rstd(x)) + b) [+ c]), rstd(x) = 1 / sqrt(mean(x^2) + eps).

    a, b, c, out: [M, N] with unit column stride; x: [M, K] with unit column stride,
    the pre-norm input of the GEMM that produced a. out defaults to a new [M, N]
    tensor of b's dtype.
    """
    num_rows, n = a.shape
    k = x.shape[1]
    if out is None:
        out = torch.empty((num_rows, n), dtype=b.dtype, device=a.device)
    if num_rows == 0:
        return out
    # MI355X sweep of 512/1024/2048 columns x 2/4/8 warps at K3's [M, 7168] tail:
    # 1024 x 4 is best at 80 rows and within 0.2 us of the best at 12 and 300.
    block_n = min(1024, triton.next_power_of_2(n))
    _rstd_scale_add3_kernel[(num_rows, triton.cdiv(n, block_n))](
        a,
        a.stride(0),
        x,
        x.stride(0),
        b,
        b.stride(0),
        c if c is not None else b,
        c.stride(0) if c is not None else 0,
        out,
        out.stride(0),
        float(eps),
        N=n,
        K=k,
        K_POW2=triton.next_power_of_2(k),
        BLOCK_N=block_n,
        HAS_C=c is not None,
        num_warps=4,
    )
    return out
