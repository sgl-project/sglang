"""Fused Kimi-K3 latent-MoE tail for small decode batches on gfx950.

One launch for what the TP8 latent MoE otherwise runs as three (RMSNorm, the
3584 -> 7168 up-projection, and the add3 with the shared-expert output and the
residual prefix):
    proj = bf16(inv_rms(latent) * (bf16(latent * norm_weight) @ up_weight.T))
    out  = bf16(proj + shared [+ prefix])                         # fp32 add
The per-row 1/rms commutes with the GEMM, so it is accumulated in the same pass
over the latent and applied in the epilogue; the GEMM input is rounded to bf16
before the row scale instead of after it, within bf16 noise of the unfused chain.

Decode M is tiny, so this is a skinny GEMM bound by streaming up_weight once:
each program owns BLOCK_N output columns for all M rows and walks K in full,
with the weight tile as the MFMA A operand in its natural [N, K] layout.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Largest decode batch this path is tuned for; larger M keeps the unfused chain,
# which is faster from M=48 on MI350X with HBM-streamed weights.
MAX_TOKENS: int = 32


@triton.jit
def _latent_tail_kernel(
    latent_ptr,  # [M, L] bf16
    norm_w_ptr,  # [L]
    w_ptr,  # [H, L] bf16
    shared_ptr,  # [M, H] bf16
    prefix_ptr,  # [M, H] bf16
    out_ptr,  # [M, H] bf16
    M,
    eps,
    stride_lm,
    stride_wn,
    stride_sm,
    stride_pm,
    stride_om,
    L: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < H

    # acc[n, m] = sum_k W[n, k] * bf16(x[m, k] * nw[k]); sq[m] = sum_k x[m, k]^2
    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in tl.range(0, L, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < L
        xt = tl.load(
            latent_ptr + offs_m[None, :] * stride_lm + offs_k[:, None],
            mask=mask_m[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        nw = tl.load(norm_w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        # The weight is streamed once; skip L1 (".cg") so it does not evict x.
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        sq += tl.sum(xt * xt, axis=0)
        acc += tl.dot(w, (xt * nw[:, None]).to(tl.bfloat16))

    inv_rms = 1.0 / tl.sqrt(sq / L + eps)
    out = (acc * inv_rms[None, :]).to(tl.bfloat16).to(tl.float32)
    mask = mask_m[None, :] & mask_n[:, None]
    out += tl.load(
        shared_ptr + offs_m[None, :] * stride_sm + offs_n[:, None], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_PREFIX:
        out += tl.load(
            prefix_ptr + offs_m[None, :] * stride_pm + offs_n[:, None],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        out_ptr + offs_m[None, :] * stride_om + offs_n[:, None],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# The column-parallel tail's split of the latent dim: 3584 = 7 x 512, one chunk per program.
PARTIAL_SPLITS: int = 7


@triton.jit
def _latent_tail_partial_kernel(
    latent_ptr,  # [M, L] bf16
    norm_w_ptr,  # [L]
    w_ptr,  # [N, L] bf16
    part_ptr,  # [SPLITS, M, N] fp32
    sq_ptr,  # [SPLITS, M] fp32
    M,
    stride_lm,
    stride_wn,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One K chunk of the tail's GEMM for BLOCK_N columns; programs of column
    block 0 also store the chunk's sum of squares of the latent rows."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    xt = tl.load(
        latent_ptr + offs_m[None, :] * stride_lm + offs_k[:, None],
        mask=mask_m[None, :],
        other=0.0,
    ).to(tl.float32)
    nw = tl.load(norm_w_ptr + offs_k).to(tl.float32)
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
        mask=(offs_n < N)[:, None],
        other=0.0,
        cache_modifier=".cg",
    )
    acc = tl.dot(w, (xt * nw[:, None]).to(tl.bfloat16))
    tl.store(
        part_ptr + (pid_k * M + offs_m[None, :]) * N + offs_n[:, None],
        acc,
        mask=mask_m[None, :] & (offs_n < N)[:, None],
    )
    if pid_n == 0:
        tl.store(sq_ptr + pid_k * M + offs_m, tl.sum(xt * xt, axis=0), mask=mask_m)


def k3_latent_tail_partials(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    # measured on MI350X (N = 896, HBM weights) against 16
    block_n: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The tail GEMM of up_weight's [N, L] rows, split over L: per chunk k,
    part[k] = bf16(latent * norm_weight)[:, chunk] @ up_weight[:, chunk].T
    and sq[k] = sum(latent[:, chunk] ** 2). The consumer finishes the tail:
    bf16(sum_k part / rms) + shared + prefix (see attention/k3_attn_res_read_hip)."""
    M, L = latent.shape
    N = up_weight.shape[0]
    assert L % PARTIAL_SPLITS == 0
    part = torch.empty(
        (PARTIAL_SPLITS, M, N), dtype=torch.float32, device=latent.device
    )
    sq = torch.empty((PARTIAL_SPLITS, M), dtype=torch.float32, device=latent.device)
    _latent_tail_partial_kernel[(triton.cdiv(N, block_n), PARTIAL_SPLITS)](
        latent,
        norm_weight,
        up_weight,
        part,
        sq,
        M,
        latent.stride(0),
        up_weight.stride(0),
        N=N,
        BLOCK_M=max(16, triton.next_power_of_2(M)),
        BLOCK_N=block_n,
        BLOCK_K=L // PARTIAL_SPLITS,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return part, sq


def supports_latent_tail(
    latent: torch.Tensor,
    up_weight: torch.Tensor,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
) -> bool:
    return (
        0 < latent.shape[0] <= MAX_TOKENS
        and latent.dtype == up_weight.dtype == shared.dtype == torch.bfloat16
        and latent.stride(-1) == 1
        and up_weight.is_contiguous()
        and shared.stride(-1) == 1
        and (
            prefix is None
            or (prefix.dtype == torch.bfloat16 and prefix.stride(-1) == 1)
        )
    )


def k3_latent_tail(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_weight: torch.Tensor,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None,
    block_n: int = 32,
    block_k: int = 512,
) -> torch.Tensor:
    """[prefix +] shared + rmsnorm(latent) @ up_weight.T for M <= MAX_TOKENS rows."""
    M, L = latent.shape
    H = up_weight.shape[0]
    if out is None:
        out = torch.empty((M, H), dtype=shared.dtype, device=latent.device)
    # Measured on MI350X at the TP8 K3 shape (L=3584, H=7168), weights from HBM.
    num_warps, waves_per_eu = (2, 2) if M == 1 else (4, 0)
    _latent_tail_kernel[(triton.cdiv(H, block_n),)](
        latent,
        norm_weight,
        up_weight,
        shared,
        shared if prefix is None else prefix,
        out,
        M,
        eps,
        latent.stride(0),
        up_weight.stride(0),
        shared.stride(0),
        0 if prefix is None else prefix.stride(0),
        out.stride(0),
        L=L,
        H=H,
        BLOCK_M=max(16, triton.next_power_of_2(M)),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_PREFIX=prefix is not None,
        num_warps=num_warps,
        num_stages=3,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
    )
    return out
