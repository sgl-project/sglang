"""BF16 out[M, N] = x[M, K] @ weight[N, K].T for decode-sized M (<= 16) and large N.

A weight-streaming kernel for wide, bandwidth-bound projections such as a vocab-parallel
LM head: each CTA owns BN output columns over the whole K, so there is no split-K
reduction. A row's arithmetic (BK tiling, FP32 accumulation order) depends only on
(N, K), never on M, so each row gives the same bits alone or in any batch of up to 16.
"""

import torch
import triton
import triton.language as tl

MAX_M = 16
_BLOCK_N = 64
_BLOCK_K = 128


@triton.jit
def _bf16_skinny_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    # int64 row offsets: a large weight or row stride exceeds 2**31 elements.
    offs_m = tl.arange(0, BM).to(tl.int64)
    offs_n = pid.to(tl.int64) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(
            w_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_first"
        )
        acc = tl.dot(a, tl.trans(w), acc)
        x_ptrs += BK
        w_ptrs += BK
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def bf16_skinny_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """1 <= M <= 16 BF16 rows on one CUDA (not ROCm) device, K % 128 == 0, unit
    stride along K and arbitrary row strides."""
    return (
        x.is_cuda
        and torch.version.hip is None
        and weight.device == x.device
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_M
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % _BLOCK_K == 0
        and x.stride(1) == 1
        and weight.stride(1) == 1
    )


def bf16_skinny_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert bf16_skinny_supported(x, weight), (
        x.shape,
        x.dtype,
        x.device,
        weight.shape,
        weight.dtype,
        weight.device,
    )
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    # Triton launches on the current device; the operands may be on another.
    with torch.cuda.device(x.device):
        _bf16_skinny_kernel[(triton.cdiv(n, _BLOCK_N),)](
            x,
            weight,
            out,
            m,
            n,
            k,
            x.stride(0),
            weight.stride(0),
            out.stride(0),
            BM=MAX_M,
            BN=_BLOCK_N,
            BK=_BLOCK_K,
            num_warps=4,
            num_stages=4,
        )
    return out
