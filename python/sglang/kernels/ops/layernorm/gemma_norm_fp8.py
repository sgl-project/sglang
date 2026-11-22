"""Gemma (residual add +) RMSNorm that writes the bf16 output and its static
per-tensor FP8 codes in one pass.

Qwen3.5 GDN layers feed the input norm to an FP8 ``in_proj_qkvz`` and a bf16
``in_proj_ba``. The unfused chain runs ``gemma_fused_add_rmsnorm`` and then
``static_quant_fp8`` on its output; this kernel replaces both.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _gemma_rmsnorm_bf16_fp8_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    scale_ptr,
    y_ptr,
    q_ptr,
    stride_x,
    stride_r,
    stride_y,
    stride_q,
    N,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r_ptrs = residual_ptr + row * stride_r + cols
        x += tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
        # Like gemma_fused_add_rmsnorm: store the rounded sum, normalize the fp32 one.
        tl.store(r_ptrs, x.to(residual_ptr.dtype.element_ty), mask=mask)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * (1.0 + w)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)

    # Static per-tensor FP8 as static_quant_fp8 on the bf16 output.
    y_s_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    y_q = tl.clamp(y.to(tl.float32) * y_s_inv, -448.0, 448.0)
    tl.store(q_ptr + row * stride_q + cols, y_q.to(q_ptr.dtype.element_ty), mask=mask)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def gemma_rmsnorm_bf16_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(y, q)``: ``y`` is ``x`` (+ ``residual``) Gemma-RMS-normalized
    with ``1 + weight`` in ``x.dtype``; ``q`` is ``y`` quantized to FP8 E4M3
    with the per-tensor ``scale``. With ``residual``, ``residual`` becomes
    ``x + residual`` and ``y`` is written over ``x``, as
    ``gemma_fused_add_rmsnorm`` does; without it ``y`` is a new tensor, as
    ``gemma_rmsnorm`` returns (the caller may still read ``x``). ``x`` must be
    2D with a contiguous last dimension."""
    assert x.dim() == 2 and x.stride(-1) == 1
    assert scale.numel() == 1
    m, n = x.shape
    q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    y = x if residual is not None else torch.empty_like(x)
    if m == 0:
        return y, q
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(-1) == 1
    block_n = triton.next_power_of_2(n)
    num_warps = min(max(block_n // 256, 1), 8)
    pdl_kwargs = {"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _gemma_rmsnorm_bf16_fp8_kernel[(m,)](
        x,
        residual if residual is not None else x,
        weight,
        scale,
        y,
        q,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        y.stride(0),
        q.stride(0),
        n,
        eps,
        HAS_RESIDUAL=residual is not None,
        BLOCK_N=block_n,
        num_warps=num_warps,
        **pdl_kwargs,
    )
    return y, q
