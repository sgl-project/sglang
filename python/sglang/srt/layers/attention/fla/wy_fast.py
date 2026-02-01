# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.utils import IS_GLUON_SUPPORTED

if IS_GLUON_SUPPORTED:
    try:
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
        from sglang.srt.layers.attention.fla.gluon.wy_fast_gluon import recompute_w_u_fwd_kernel_gluon
    except ImportError as e:
        raise ImportError(
            f">>> Failed to import Gluon in current triton version {triton.__version__} and "
            f">>> Platform {torch.cuda.get_device_capability()}.\n"
            f">>> Gluon/Blackwell features require: \n"
            f">>> 1. Triton >= 3.6.0.\n"
            f">>> 2. NVIDIA GPU (compute capability >= 10.0)\n"
            f">>> Error: {e}\n"
            f">>> Set FLA_USE_GLUON=0 to disable and continue."
        ) from e

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)

    if IS_GLUON_SUPPORTED:
        # tma desc init
        kw_layout = gl.NVMMASharedLayout.get_default_for([1, BT, 1, BK], gl.float16)
        vu_layout = gl.NVMMASharedLayout.get_default_for([1, BT, 1, BV], gl.float16)
        A_layout = gl.NVMMASharedLayout.get_default_for([1, BT, 1, BT], gl.float16) 
        k_desc = TensorDescriptor.from_tensor(k, [1, BT, 1, BK], kw_layout)
        v_desc = TensorDescriptor.from_tensor(v, [1, BT, 1, BV], vu_layout)
        A_desc = TensorDescriptor.from_tensor(A, [1, BT, 1, BT], A_layout)
        if cu_seqlens is not None:
            w_layout = gl.NVMMASharedLayout.get_default_for([BT, BK], gl.float16)
            w_desc = TensorDescriptor.from_tensor(w.view(B*T, H*K), [1, BK], w_layout)
            u_layout = gl.NVMMASharedLayout.get_default_for([BT, BV], gl.float16)
            u_desc = TensorDescriptor.from_tensor(u.view(B*T, H*V), [1, BV], u_layout)
        else:
            w_desc = TensorDescriptor.from_tensor(w, [1, BT, 1, BK], kw_layout)
            u_desc = TensorDescriptor.from_tensor(u, [1, BT, 1, BV], vu_layout)
        
        recompute_w_u_fwd_kernel_gluon[(NT, B*H)](
            k_desc=k_desc,
            v_desc=v_desc,
            w_desc=w_desc,
            u_desc=u_desc,
            A_desc=A_desc,
            beta=beta,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HK=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_G=g_cumsum is not None,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
        )
    else:
        recompute_w_u_fwd_kernel[(NT, B * H)](
            k=k,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            IS_VARLEN=cu_seqlens is not None,
            num_warps=4,
            num_stages=3,
        )
    return w, u


fwd_recompute_w_u = recompute_w_u_fwd
