"""SiLU-and-mul fused with aiter's 1x128 fp8 activation quant (ROCm)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_fp8_group_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    D,
    stride_xm,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
):
    m = tl.program_id(0)
    g0 = tl.program_id(1) * GROUPS_PER_PROG
    g = g0 + tl.arange(0, GROUPS_PER_PROG)
    cols = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    mask = cols < D
    row = x_ptr + m * stride_xm
    gate = tl.load(row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row + D + cols, mask=mask, other=0.0).to(tl.float32)
    # Round like the unfused SiluAndMul (silu_and_mul): silu(gate) to bf16, then
    # the product to bf16, which is what the unfused path quantizes.
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    a = (silu * up).to(tl.bfloat16).to(tl.float32)
    # Same scale as aiter dynamic_per_group_scaled_quant.
    scale = tl.maximum(tl.max(tl.abs(a), axis=1), 1e-10) * (1.0 / FP8_MAX)
    q = tl.clamp(a * (1.0 / scale)[:, None], -FP8_MAX, FP8_MAX)
    tl.store(q_ptr + m * D + cols, q.to(q_ptr.dtype.element_ty), mask=mask)
    g_mask = g * GROUP < D
    if TRANSPOSE_SCALE:
        tl.store(s_ptr + g * M + m, scale, mask=g_mask)
    else:
        tl.store(s_ptr + m * tl.cdiv(D, GROUP) + g, scale, mask=g_mask)


def silu_mul_fp8_group_quant(
    x: torch.Tensor, transpose_scale: bool, group_size: int = 128
):
    """``x`` is [M, 2 * D] (gate | up). Returns fp8 [M, D] and its fp32 group
    scale [M, D / group_size]. With ``transpose_scale`` the scale bytes are in
    [groups, M] order behind a row-major-shaped tensor, like aiter quant's
    ``transpose_scale=True``."""
    M, D2 = x.shape
    D = D2 // 2
    assert D % group_size == 0 and x.stride(1) == 1
    dtype = torch.float8_e4m3fn
    q = torch.empty((M, D), dtype=dtype, device=x.device)
    s = torch.empty((M, D // group_size), dtype=torch.float32, device=x.device)
    if M == 0:
        return q, s
    groups_per_prog = 8
    grid = (M, triton.cdiv(D // group_size, groups_per_prog))
    _silu_mul_fp8_group_quant_kernel[grid](
        x,
        q,
        s,
        M,
        D,
        x.stride(0),
        FP8_MAX=torch.finfo(dtype).max,
        GROUP=group_size,
        GROUPS_PER_PROG=groups_per_prog,
        TRANSPOSE_SCALE=transpose_scale,
        num_warps=4,
    )
    return q, s
