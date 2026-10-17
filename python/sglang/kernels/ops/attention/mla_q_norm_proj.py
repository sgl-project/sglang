"""MLA query normalization and projection in one launch (decode / verify batches).

For the absorbed-MLA path's q_a_layernorm + kv_a_layernorm + q_b_proj:
    q  = bf16(bf16(rmsnorm(q_a) * w_q) @ w_qb.T)     [M, N]
    kv = bf16(rmsnorm(kv_a) * w_kv)                    [M, KV]
Programs before the last each own BLOCK_N output columns and recompute the q_a row
statistics (q_a is a few KB per row), so the normalized activation never goes to
memory; the last program normalizes kv_a. The norms round once, as aiter's
fused_qk_rmsnorm does, and the projection accumulates in fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS: int = 16
# (BLOCK_N, BLOCK_K, num_warps, num_stages) by rows: the fastest of a sweep on MI350X
# over the Kimi-K3 TP8 shapes.
_CONFIG_ONE_ROW = (16, 512, 1, 3)
_CONFIG = (16, 256, 8, 3)


@triton.jit
def _mla_q_norm_proj_kernel(
    q_ptr,
    kv_ptr,
    q_norm_ptr,
    kv_norm_ptr,
    w_ptr,
    out_ptr,
    kv_out_ptr,
    M,
    stride_q,
    stride_kv,
    stride_out,
    stride_kv_out,
    q_eps,
    kv_eps,
    QA: tl.constexpr,
    KV: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M
    if pid == N // BLOCK_N:
        offs = tl.arange(0, KV)
        x = tl.load(
            kv_ptr + offs_m[:, None] * stride_kv + offs[None, :], mask=mask_m, other=0.0
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / KV + kv_eps)
        w = tl.load(kv_norm_ptr + offs).to(tl.float32)
        y = x * rstd[:, None] * w[None, :]
        tl.store(
            kv_out_ptr + offs_m[:, None] * stride_kv_out + offs[None, :],
            y.to(tl.bfloat16),
            mask=mask_m,
        )
        return
    offs_k = tl.arange(0, BLOCK_K)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, QA, BLOCK_K):
        x = tl.load(
            q_ptr + offs_m[:, None] * stride_q + (k + offs_k)[None, :],
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        sq += tl.sum(x * x, axis=1)
    rstd = tl.rsqrt(sq / QA + q_eps)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, QA, BLOCK_K):
        x = tl.load(
            q_ptr + offs_m[:, None] * stride_q + (k + offs_k)[None, :],
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        w_norm = tl.load(q_norm_ptr + k + offs_k).to(tl.float32)
        xn = (x * rstd[:, None] * w_norm[None, :]).to(tl.bfloat16)
        w = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * QA + (k + offs_k)[:, None],
            cache_modifier=".cg",
        )
        acc = tl.dot(xn, w, acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m,
    )


def covered(q_a: torch.Tensor, kv_a: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        0 < q_a.shape[0] <= MAX_ROWS
        and q_a.dtype == kv_a.dtype == weight.dtype == torch.bfloat16
        and q_a.stride(-1) == 1
        and kv_a.stride(-1) == 1
        and weight.is_contiguous()
        and weight.shape[1] == q_a.shape[1]
        and weight.shape[0] % 16 == 0
        and q_a.shape[1] % 512 == 0
        and triton.next_power_of_2(kv_a.shape[1]) == kv_a.shape[1]
    )


def mla_q_norm_proj(
    q_a: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_eps: float,
    kv_a: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    kv_eps: float,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(q [M, N], normalized kv_a [M, KV]) for q_a [M, QA], kv_a [M, KV] (row strides
    free) and the q_b weight [N, QA]; the caller checked covered()."""
    m, qa = q_a.shape
    n, kv = weight.shape[0], kv_a.shape[1]
    out = q_a.new_empty((m, n))
    kv_out = kv_a.new_empty((m, kv))
    block_n, block_k, num_warps, num_stages = _CONFIG_ONE_ROW if m == 1 else _CONFIG
    _mla_q_norm_proj_kernel[(n // block_n + 1,)](
        q_a,
        kv_a,
        q_norm_weight,
        kv_norm_weight,
        weight,
        out,
        kv_out,
        m,
        q_a.stride(0),
        kv_a.stride(0),
        out.stride(0),
        kv_out.stride(0),
        q_eps,
        kv_eps,
        QA=qa,
        KV=kv,
        N=n,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, kv_out
