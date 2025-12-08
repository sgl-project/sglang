# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _fused_cumsum_kkt_kernel(
    g_ptr, k_ptr, beta_ptr,
    g_cumsum_ptr, A_ptr,
    cu_seqlens, chunk_indices,
    T, H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_seq = eos - bos
        i_t = i_t_local
    else:
        bos = i_b * T
        T_seq = T
    
    o_t = tl.arange(0, BT)
    
    p_g = tl.make_block_ptr(g_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_cumsum = tl.cumsum(b_g, axis=0)
    p_g_out = tl.make_block_ptr(g_cumsum_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_g_out, b_g_cumsum.to(p_g_out.dtype.element_ty), boundary_check=(0,))
    
    p_beta = tl.make_block_ptr(beta_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    
    p_k = tl.make_block_ptr(k_ptr + (bos * Hg + i_h // (H // Hg)) * K, (T_seq, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    
    b_A = tl.dot(b_k, tl.trans(b_k))
    b_g_diff = b_g_cumsum[:, None] - b_g_cumsum[None, :]
    b_A = b_A * safe_exp(b_g_diff) * b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0.0)
    
    p_A = tl.make_block_ptr(A_ptr + (bos * H + i_h) * BT, (T_seq, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(A_ptr.dtype.element_ty), boundary_check=(0, 1))


def fused_cumsum_kkt(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused cumsum + KKT.

    Args:
        g: [B, T, H]
        k: [B, T, Hg, K]
        beta: [B, T, H]

    Returns:
        g_cumsum: [B, T, H]
        A: [B, T, H, chunk_size], strictly lower triangular
    """
    B, T, H = g.shape
    Hg, K = k.shape[2], k.shape[3]
    
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)
    
    g_cumsum = torch.empty(B, T, H, device=g.device, dtype=torch.float32)
    A = torch.empty(B, T, H, chunk_size, device=k.device, dtype=torch.float32)
    
    _fused_cumsum_kkt_kernel[(NT, B * H)](
        g, k, beta, g_cumsum, A,
        cu_seqlens, chunk_indices,
        T, H, Hg, K, chunk_size,
        num_warps=4, num_stages=3
    )
    return g_cumsum, A
