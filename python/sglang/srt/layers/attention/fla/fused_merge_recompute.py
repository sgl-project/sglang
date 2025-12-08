# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _fused_merge_recompute_kernel(
    k_ptr, v_ptr, beta_ptr,
    g_cumsum_ptr, A_ptr, Ai16_ptr, w_ptr, u_ptr,
    cu_seqlens, chunk_indices,
    T, H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, IS_VARLEN: tl.constexpr,
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
    
    o_16 = tl.arange(0, 16)
    
    Ai16_base = Ai16_ptr + (bos * H + i_h) * 16
    p_Ai11 = tl.make_block_ptr(Ai16_base, (T_seq, 16), (H * 16, 1), (i_t * 64, 0), (16, 16), (1, 0))
    p_Ai22 = tl.make_block_ptr(Ai16_base, (T_seq, 16), (H * 16, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_Ai33 = tl.make_block_ptr(Ai16_base, (T_seq, 16), (H * 16, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_Ai44 = tl.make_block_ptr(Ai16_base, (T_seq, 16), (H * 16, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    
    b_Ai11 = tl.load(p_Ai11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Ai22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Ai33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai44 = tl.load(p_Ai44, boundary_check=(0, 1)).to(tl.float32)
    
    A_base = A_ptr + (bos * H + i_h) * BT
    p_A21 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 16, 0), (16, 16), (1, 0))
    p_A31 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 32, 0), (16, 16), (1, 0))
    p_A32 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 32, 16), (16, 16), (1, 0))
    p_A41 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 48, 0), (16, 16), (1, 0))
    p_A42 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 48, 16), (16, 16), (1, 0))
    p_A43 = tl.make_block_ptr(A_base, (T_seq, BT), (H * BT, 1), (i_t * 64 + 48, 32), (16, 16), (1, 0))
    
    b_A21 = tl.load(p_A21, boundary_check=(0, 1)).to(tl.float32)
    b_A31 = tl.load(p_A31, boundary_check=(0, 1)).to(tl.float32)
    b_A32 = tl.load(p_A32, boundary_check=(0, 1)).to(tl.float32)
    b_A41 = tl.load(p_A41, boundary_check=(0, 1)).to(tl.float32)
    b_A42 = tl.load(p_A42, boundary_check=(0, 1)).to(tl.float32)
    b_A43 = tl.load(p_A43, boundary_check=(0, 1)).to(tl.float32)
    
    b_Ai21 = -tl.dot(tl.dot(b_Ai22, b_A21, input_precision="ieee"), b_Ai11, input_precision="ieee")
    b_Ai32 = -tl.dot(tl.dot(b_Ai33, b_A32, input_precision="ieee"), b_Ai22, input_precision="ieee")
    b_Ai43 = -tl.dot(tl.dot(b_Ai44, b_A43, input_precision="ieee"), b_Ai33, input_precision="ieee")
    b_Ai31 = -tl.dot(b_Ai33, tl.dot(b_A31, b_Ai11, input_precision="ieee") + tl.dot(b_A32, b_Ai21, input_precision="ieee"), input_precision="ieee")
    b_Ai42 = -tl.dot(b_Ai44, tl.dot(b_A42, b_Ai22, input_precision="ieee") + tl.dot(b_A43, b_Ai32, input_precision="ieee"), input_precision="ieee")
    b_Ai41 = -tl.dot(b_Ai44, tl.dot(b_A41, b_Ai11, input_precision="ieee") + tl.dot(b_A42, b_Ai21, input_precision="ieee") + tl.dot(b_A43, b_Ai31, input_precision="ieee"), input_precision="ieee")
    
    k_base = k_ptr + (bos * Hg + i_h // (H // Hg)) * K
    beta_base = beta_ptr + bos * H + i_h
    g_base = g_cumsum_ptr + bos * H + i_h
    
    p_k1 = tl.make_block_ptr(k_base, (T_seq, K), (Hg * K, 1), (i_t * 64, 0), (16, K), (1, 0))
    p_k2 = tl.make_block_ptr(k_base, (T_seq, K), (Hg * K, 1), (i_t * 64 + 16, 0), (16, K), (1, 0))
    p_k3 = tl.make_block_ptr(k_base, (T_seq, K), (Hg * K, 1), (i_t * 64 + 32, 0), (16, K), (1, 0))
    p_k4 = tl.make_block_ptr(k_base, (T_seq, K), (Hg * K, 1), (i_t * 64 + 48, 0), (16, K), (1, 0))
    
    b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
    b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
    b_k4 = tl.load(p_k4, boundary_check=(0, 1)).to(tl.float32)
    
    p_beta1 = tl.make_block_ptr(beta_base, (T_seq,), (H,), (i_t * 64,), (16,), (0,))
    p_beta2 = tl.make_block_ptr(beta_base, (T_seq,), (H,), (i_t * 64 + 16,), (16,), (0,))
    p_beta3 = tl.make_block_ptr(beta_base, (T_seq,), (H,), (i_t * 64 + 32,), (16,), (0,))
    p_beta4 = tl.make_block_ptr(beta_base, (T_seq,), (H,), (i_t * 64 + 48,), (16,), (0,))
    
    b_beta1 = tl.load(p_beta1, boundary_check=(0,)).to(tl.float32)
    b_beta2 = tl.load(p_beta2, boundary_check=(0,)).to(tl.float32)
    b_beta3 = tl.load(p_beta3, boundary_check=(0,)).to(tl.float32)
    b_beta4 = tl.load(p_beta4, boundary_check=(0,)).to(tl.float32)
    
    p_g1 = tl.make_block_ptr(g_base, (T_seq,), (H,), (i_t * 64,), (16,), (0,))
    p_g2 = tl.make_block_ptr(g_base, (T_seq,), (H,), (i_t * 64 + 16,), (16,), (0,))
    p_g3 = tl.make_block_ptr(g_base, (T_seq,), (H,), (i_t * 64 + 32,), (16,), (0,))
    p_g4 = tl.make_block_ptr(g_base, (T_seq,), (H,), (i_t * 64 + 48,), (16,), (0,))
    
    b_g1 = tl.exp(tl.load(p_g1, boundary_check=(0,)).to(tl.float32))
    b_g2 = tl.exp(tl.load(p_g2, boundary_check=(0,)).to(tl.float32))
    b_g3 = tl.exp(tl.load(p_g3, boundary_check=(0,)).to(tl.float32))
    b_g4 = tl.exp(tl.load(p_g4, boundary_check=(0,)).to(tl.float32))
    
    b_rhs_w1 = b_k1 * b_beta1[:, None] * b_g1[:, None]
    b_rhs_w2 = b_k2 * b_beta2[:, None] * b_g2[:, None]
    b_rhs_w3 = b_k3 * b_beta3[:, None] * b_g3[:, None]
    b_rhs_w4 = b_k4 * b_beta4[:, None] * b_g4[:, None]
    
    b_w1 = tl.dot(b_Ai11, b_rhs_w1, input_precision="ieee")
    b_w2 = tl.dot(b_Ai21, b_rhs_w1, input_precision="ieee") + tl.dot(b_Ai22, b_rhs_w2, input_precision="ieee")
    b_w3 = tl.dot(b_Ai31, b_rhs_w1, input_precision="ieee") + tl.dot(b_Ai32, b_rhs_w2, input_precision="ieee") + tl.dot(b_Ai33, b_rhs_w3, input_precision="ieee")
    b_w4 = tl.dot(b_Ai41, b_rhs_w1, input_precision="ieee") + tl.dot(b_Ai42, b_rhs_w2, input_precision="ieee") + tl.dot(b_Ai43, b_rhs_w3, input_precision="ieee") + tl.dot(b_Ai44, b_rhs_w4, input_precision="ieee")
    
    w_base = w_ptr + (bos * H + i_h) * K
    p_w1 = tl.make_block_ptr(w_base, (T_seq, K), (H * K, 1), (i_t * 64, 0), (16, K), (1, 0))
    p_w2 = tl.make_block_ptr(w_base, (T_seq, K), (H * K, 1), (i_t * 64 + 16, 0), (16, K), (1, 0))
    p_w3 = tl.make_block_ptr(w_base, (T_seq, K), (H * K, 1), (i_t * 64 + 32, 0), (16, K), (1, 0))
    p_w4 = tl.make_block_ptr(w_base, (T_seq, K), (H * K, 1), (i_t * 64 + 48, 0), (16, K), (1, 0))
    tl.store(p_w1, b_w1.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_w2, b_w2.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_w3, b_w3.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_w4, b_w4.to(w_ptr.dtype.element_ty), boundary_check=(0, 1))
    
    v_base = v_ptr + (bos * H + i_h) * V
    p_v1 = tl.make_block_ptr(v_base, (T_seq, V), (H * V, 1), (i_t * 64, 0), (16, V), (1, 0))
    p_v2 = tl.make_block_ptr(v_base, (T_seq, V), (H * V, 1), (i_t * 64 + 16, 0), (16, V), (1, 0))
    p_v3 = tl.make_block_ptr(v_base, (T_seq, V), (H * V, 1), (i_t * 64 + 32, 0), (16, V), (1, 0))
    p_v4 = tl.make_block_ptr(v_base, (T_seq, V), (H * V, 1), (i_t * 64 + 48, 0), (16, V), (1, 0))
    
    b_v1 = tl.load(p_v1, boundary_check=(0, 1)).to(tl.float32)
    b_v2 = tl.load(p_v2, boundary_check=(0, 1)).to(tl.float32)
    b_v3 = tl.load(p_v3, boundary_check=(0, 1)).to(tl.float32)
    b_v4 = tl.load(p_v4, boundary_check=(0, 1)).to(tl.float32)
    
    b_rhs_u1 = b_v1 * b_beta1[:, None]
    b_rhs_u2 = b_v2 * b_beta2[:, None]
    b_rhs_u3 = b_v3 * b_beta3[:, None]
    b_rhs_u4 = b_v4 * b_beta4[:, None]
    
    b_u1 = tl.dot(b_Ai11, b_rhs_u1, input_precision="ieee")
    b_u2 = tl.dot(b_Ai21, b_rhs_u1, input_precision="ieee") + tl.dot(b_Ai22, b_rhs_u2, input_precision="ieee")
    b_u3 = tl.dot(b_Ai31, b_rhs_u1, input_precision="ieee") + tl.dot(b_Ai32, b_rhs_u2, input_precision="ieee") + tl.dot(b_Ai33, b_rhs_u3, input_precision="ieee")
    b_u4 = tl.dot(b_Ai41, b_rhs_u1, input_precision="ieee") + tl.dot(b_Ai42, b_rhs_u2, input_precision="ieee") + tl.dot(b_Ai43, b_rhs_u3, input_precision="ieee") + tl.dot(b_Ai44, b_rhs_u4, input_precision="ieee")
    
    u_base = u_ptr + (bos * H + i_h) * V
    p_u1 = tl.make_block_ptr(u_base, (T_seq, V), (H * V, 1), (i_t * 64, 0), (16, V), (1, 0))
    p_u2 = tl.make_block_ptr(u_base, (T_seq, V), (H * V, 1), (i_t * 64 + 16, 0), (16, V), (1, 0))
    p_u3 = tl.make_block_ptr(u_base, (T_seq, V), (H * V, 1), (i_t * 64 + 32, 0), (16, V), (1, 0))
    p_u4 = tl.make_block_ptr(u_base, (T_seq, V), (H * V, 1), (i_t * 64 + 48, 0), (16, V), (1, 0))
    tl.store(p_u1, b_u1.to(u_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_u2, b_u2.to(u_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_u3, b_u3.to(u_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_u4, b_u4.to(u_ptr.dtype.element_ty), boundary_check=(0, 1))


def fused_merge_recompute(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    Ai16: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused merge + recompute.

    Args:
        k: [B, T, Hg, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g_cumsum: [B, T, H]
        A: [B, T, H, chunk_size]
        Ai16: [B, T, H, 16], diagonal block inverses from solve_tril_16x16

    Returns:
        w: [B, T, H, K]
        u: [B, T, H, V]
    """
    B, T, H = g_cumsum.shape
    Hg, K = k.shape[2], k.shape[3]
    V = v.shape[3]
    
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)
    
    w = torch.empty(B, T, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    
    _fused_merge_recompute_kernel[(NT, B * H)](
        k, v, beta, g_cumsum, A, Ai16, w, u,
        cu_seqlens, chunk_indices,
        T, H, Hg, K, V, chunk_size,
        num_warps=4, num_stages=2,
    )
    
    return w, u
