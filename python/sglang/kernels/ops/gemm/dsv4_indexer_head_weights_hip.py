from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k


@triton.jit
def _reduce_scale_bf16_block(
    block,
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    scale,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One BLOCK of _reduce_scale_bf16_kernel (shared with the fused index-Q launch)."""
    offs = block * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = m < M
    acc = tl.load(part_ptr + m * stride_pm + n, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_ps + m * stride_pm + n, mask=mask, other=0.0
        )
    # bf16(bf16(sum) * scale): the linear's bf16 output, then the aten multiply rounded to bf16
    w = acc.to(tl.bfloat16).to(tl.float32) * scale
    tl.store(out_ptr + m * stride_om + n, w.to(tl.bfloat16), mask=mask)


@triton.jit
def _reduce_scale_bf16_kernel(
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    scale,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _reduce_scale_bf16_block(
        tl.program_id(0),
        part_ptr,
        out_ptr,
        M,
        stride_ps,
        stride_pm,
        stride_om,
        scale,
        N=N,
        SPLIT_K=SPLIT_K,
        BLOCK=BLOCK,
    )


def indexer_head_weights(
    x: torch.Tensor, weight: torch.Tensor, scale: float
) -> torch.Tensor:
    """bf16(bf16(x @ weight.T) * scale) as a contiguous bf16 [M, N], the layout the
    FlyDSL logits kernels take. x bf16 [M, K] and weight bf16 [N, K], with
    M <= rocm_gemv_split_k_max_tokens(n=N, k=K, weight_dtype=torch.bfloat16)."""
    partials = rocm_router_gemv_split_k(x, weight)
    split_k, M, N = partials.shape
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    block = triton.next_power_of_2(M * N)
    _reduce_scale_bf16_kernel[(triton.cdiv(M * N, block),)](
        partials,
        out,
        M,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        float(scale),
        N=N,
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )
    return out
