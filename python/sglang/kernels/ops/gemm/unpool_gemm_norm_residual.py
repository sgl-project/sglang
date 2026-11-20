"""MoE unpooler GEMM fused with the FFN-boundary norms and gated residual (Muse Spark, BF16).

Two ordered launches: the BF16 split-K partials of ``expert_output @ W_unpool.T`` (``bf16_splitk_gemm``), then
one row program that sums the splits, rounds to BF16 as the GEMM store would, applies the input-scale RMSNorm,
the gated residual update ``stream = alpha * stream + beta * norm(branch)`` (FP32 stream) and the next layer's
pre-attention RMSNorm. Both BF16 rounding boundaries of the unfused sequence are kept.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials


@triton.jit
def _unpool_reduce_boundary(
    PARTIAL,
    STREAM,
    OUTPUT,
    NEW_STREAM,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    M: tl.constexpr,
    H: tl.constexpr,
    SK: tl.constexpr,
    POST_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    PRE_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, H)
    split = tl.arange(0, SK)
    partial = tl.load(PARTIAL + split[:, None] * M * H + row * H + h[None, :])
    # The original GEMM stores BF16 before the input-scale norm. Keep both
    # BF16 rounding boundaries; moving the norm into FP32 GEMM would differ.
    branch = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
    post_weight = tl.load(POST_WEIGHT + h).to(tl.float32)
    post_input = (branch * post_weight).to(tl.bfloat16).to(tl.float32)
    post_rstd = tl.rsqrt(tl.sum(post_input * post_input, axis=0) / H + POST_EPS)
    normed_branch = (post_input * post_rstd).to(tl.bfloat16).to(tl.float32)
    stream = tl.load(STREAM + row * H + h)
    alpha, beta = tl.load(ALPHA + h), tl.load(BETA + h)
    new_stream = alpha * stream + beta * normed_branch
    tl.store(NEW_STREAM + row * H + h, new_stream)
    pre_input = new_stream
    if PRE_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)
    pre_rstd = tl.rsqrt(tl.sum(pre_input * pre_input, axis=0) / H + PRE_EPS)
    pre_weight = tl.load(PRE_WEIGHT + h).to(tl.float32)
    normalized = pre_input * pre_rstd * pre_weight
    tl.store(OUTPUT + row * H + h, normalized)


def unpool_gemm_norm_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    *,
    block_n: int = 64,
    block_k: int = 128,
    split_k: int = 4,
    pre_bf16: bool = False,
    num_warps: int = 16,
):
    """Return (next layer's normalized attention input BF16 [M, H], updated FP32 stream [M, H])."""
    m, k = x.shape
    h = weight.shape[0]
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous() and stream.is_contiguous()
    assert k == 4096 and h == 8192 and 0 < m <= 64
    assert stream.dtype == torch.float32 and stream.shape == (m, h)
    assert split_k in (2, 4, 8)
    partial = torch.empty((split_k, m, h), dtype=torch.float32, device=x.device)
    output = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    new_stream = torch.empty_like(stream)
    bf16_splitk_gemm_partials(
        x, weight, partial, block_n=block_n, block_k=block_k, split_k=split_k
    )
    _unpool_reduce_boundary[(m,)](
        partial,
        stream,
        output,
        new_stream,
        post_weight,
        pre_weight,
        alpha,
        beta,
        m,
        h,
        split_k,
        post_eps,
        pre_eps,
        pre_bf16,
        num_warps=num_warps,
        num_stages=1,
    )
    return output, new_stream
