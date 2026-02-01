# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import check_shared_mem, is_nvidia_hopper, IS_GLUON_SUPPORTED

if IS_GLUON_SUPPORTED:
    try:
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
        from sglang.srt.layers.attention.fla.gluon.chunk_o_gluon import chunk_fwd_kernel_o_gluon

    except ImportError as e:
        raise ImportError(
            f">>> Failed to import Gluon in current triton version {triton.__version__} and "
            f">>> Platform {torch.cuda.get_device_capability()}.\n"
            f">>> Gluon/Blackwell features require: \n"
            f">>> 1. Triton >= 3.6.0\n"
            f">>> 2. NVIDIA GPU (compute capability >= 10.0)\n"
            f">>> Error: {e}\n"
            f">>> Set FLA_USE_GLUON=0 to disable and continue."
        ) from e


BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


# @triton.autotune(
#     configs=[
#         triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in NUM_WARPS
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)

    if IS_GLUON_SUPPORTED:
        BK = 128 if K >= 128 else 64
        BV = 128 if V >= 128 else 64
        IS_VARLEN = cu_seqlens is not None
        num_warps = 8 if BT >= 128 else 4

        gl_dtype = gl.bfloat16 if q.dtype == torch.bfloat16 else gl.float16
        qk_layout = gl.NVMMASharedLayout.get_default_for([1, BT, 1, BK], gl_dtype)
        vo_layout = gl.NVMMASharedLayout.get_default_for([1, BT, 1, BV], gl_dtype)
        h_layout = gl.NVMMASharedLayout.get_default_for([1, 1, 1, BK, BV], gl_dtype)
        q_desc = TensorDescriptor.from_tensor(q, [1, BT, 1, BK], qk_layout)
        k_desc = TensorDescriptor.from_tensor(k, [1, BT, 1, BK], qk_layout)
        v_desc = TensorDescriptor.from_tensor(v, [1, BT, 1, BV], vo_layout)
        h_desc = TensorDescriptor.from_tensor(h, [1, 1, 1, BK, BV], h_layout)
        if IS_VARLEN:
            o_layout = gl.NVMMASharedLayout.get_default_for([BT, BV], gl_dtype)
            o_desc = TensorDescriptor.from_tensor(o.view(B * T, H * V), [1, BV], o_layout)
        else:
            o_desc = TensorDescriptor.from_tensor(o, [1, BT, 1, BV], vo_layout)

        grid = (triton.cdiv(V, BV), NT, B * H)
        chunk_fwd_kernel_o_gluon[grid](
            q_desc=q_desc,
            k_desc=k_desc,
            v_desc=v_desc,
            h_desc=h_desc,
            o_desc=o_desc,
            g=g,
            g_gamma=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            HK=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_G=g is not None,
            IS_VARLEN=IS_VARLEN,
            num_warps=num_warps,
        )
    else:
        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), NT, B * H)

        chunk_fwd_kernel_o[grid](
            q,
            k,
            v,
            h,
            g,
            o,
            cu_seqlens,
            chunk_indices,
            scale,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=128,
            BV=64,
            USE_G=g is not None,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=2,
        )
    return o
