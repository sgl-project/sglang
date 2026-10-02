"""Prefill-time low-ratio indexer scores for Hopper.

The torch prefill path scores a chunk of query rows against the request's
dequantized index K as `einsum -> relu -> * weights -> sum over heads`, which
materializes a bf16 [rows, heads, lc] tensor and streams it through HBM several
times. This Triton kernel keeps the head reduction in a [BLOCK_M, BLOCK_N]
register tile and writes only the fp32 [rows, lc] result, with the length mask
already applied.

Numerics follow the torch path and the decode kernel in fp4_indexer.py
(bf16 dot, bf16 relu/weight product, bf16 head reduction), so prefill and decode
select the same positions; only the head summation order differs.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import INDEX_HEAD_DIM

# Measured best on H200 among 7 swept tiles. The grid walks query-row blocks
# outermost so a row block's queries (BLOCK_M * heads * d bf16) stay in L2
# while the index K is streamed past them.
_CONFIG = dict(BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=2)


@triton.jit
def _index_scores_kernel(
    q_ptr,  # [M, H, D] bf16
    k_ptr,  # [N, D] bf16
    w_ptr,  # [M, H] bf16
    lens_ptr,  # [M] int, visible positions per row
    out_ptr,  # [M, N] fp32, -inf at positions >= lens
    M,
    N,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_wm,
    stride_om,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // n_blocks
    pid_n = pid % n_blocks
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k = tl.load(
        k_ptr + offs_n[:, None].to(tl.int64) * stride_kn + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    k_t = tl.trans(k)
    q_rows = q_ptr + offs_m[:, None].to(tl.int64) * stride_qm + offs_d[None, :]
    w_rows = w_ptr + offs_m.to(tl.int64) * stride_wm
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        q = tl.load(q_rows + h * stride_qh, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_rows + h, mask=mask_m, other=0.0).to(tl.float32)
        # reference rounding points, as in the decode kernel:
        # bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
        s = tl.dot(q, k_t).to(tl.bfloat16).to(tl.float32)
        acc += (tl.maximum(s, 0.0) * w[:, None]).to(tl.bfloat16).to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)

    lens = tl.load(lens_ptr + offs_m, mask=mask_m, other=0)
    acc = tl.where(offs_n[None, :] < lens[:, None], acc, float("-inf"))
    tl.store(
        out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fused_index_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, lens: torch.Tensor
) -> torch.Tensor:
    """q [t, H, d] bf16, k [n, d] bf16, weights [t, H] bf16, lens [t] int ->
    [t, n] fp32: sum over heads of relu(q_h k^T) * w_h, -inf at positions >= lens."""
    t, heads, head_dim = q.shape
    n = k.shape[0]
    assert q.dtype == k.dtype == torch.bfloat16 and head_dim == INDEX_HEAD_DIM
    assert q.stride(2) == 1 and k.stride(1) == 1 and weights.stride(1) == 1
    assert lens.stride(0) == 1
    out = torch.empty(t, n, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(t, _CONFIG["BLOCK_M"]) * triton.cdiv(n, _CONFIG["BLOCK_N"]),)
    _index_scores_kernel[grid](
        q,
        k,
        weights,
        lens,
        out,
        t,
        n,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        out.stride(0),
        H=heads,
        D=head_dim,
        **_CONFIG,
    )
    return out
