"""Single-token W8A16 GEMV for 128x128 block-quantized FP8 weights.

Decode at batch size 1 streams every dense weight once per token. The block
FP8 GEMM path quantizes the bf16 activation to FP8 (one extra launch) and then
runs a tensor-core GEMM tiled for M >= 64, which at M = 1 leaves a large part
of DRAM bandwidth unused. Here each CTA owns BLOCK_N output rows over the full
K: it loads the bf16 activation once, applies the per-block weight scales to
it, and reduces fp8 weight rows in fp32. One launch, no split-K partials, no
activation quantization (which also makes it slightly more accurate than the
W8A8 path).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_GROUP = 128


@triton.jit
def _block_fp8_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    y_ptr,
    N,
    stride_wn,
    stride_sn,
    stride_sk,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    # BLOCK_N divides GROUP, so every row of this CTA shares one scale row.
    s_row = s_ptr + ((pid * BLOCK_N) // GROUP) * stride_sn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        xs = tl.load(x_ptr + rk).to(tl.float32) * tl.load(
            s_row + (rk // GROUP) * stride_sk
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :],
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * xs[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def _config(n: int, k: int) -> tuple[int, int, int]:
    """(BLOCK_N, BLOCK_K, num_warps), from a cold-L2 sweep on H100 SXM.

    About N/1024 rows per CTA (2..16) keeps the grid at ~500-1800 CTAs; each
    K step covers up to 8K weight elements, the largest power of two dividing K.
    """
    block_n = 2
    while block_n < 16 and block_n * 2 * 1024 <= n:
        block_n *= 2
    block_k = 8192 // block_n
    while k % block_k:
        block_k //= 2
    num_warps = 2 if block_n <= 4 else 4
    return block_n, block_k, num_warps


def use_block_fp8_gemv(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> bool:
    if not (
        x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == torch.float8_e4m3fn
        and weight.dim() == 2
        and weight.stride(1) == 1
        and weight_scale.dtype == torch.float32
        and weight_scale.dim() == 2
    ):
        return False
    n, k = weight.shape
    return x.numel() == k and k % _GROUP == 0 and n >= 64


def block_fp8_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y[1, N] = x[1, K] @ dequant(weight[N, K], weight_scale)^T.

    weight_scale holds one fp32 scale per 128x128 block, logical shape
    [>= ceil(N/128), >= K/128]; any strides (e.g. a TMA-aligned layout) work.
    """
    n, k = weight.shape
    block_n, block_k, num_warps = _config(n, k)
    x = x.reshape(1, k)
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty((1, n), dtype=x.dtype, device=x.device)
    _block_fp8_gemv_kernel[(triton.cdiv(n, block_n),)](
        x,
        weight,
        weight_scale,
        bias if bias is not None else y,
        y,
        n,
        weight.stride(0),
        weight_scale.stride(0),
        weight_scale.stride(1),
        K=k,
        GROUP=_GROUP,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
    )
    return y
