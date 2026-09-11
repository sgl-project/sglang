# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/model_executor/layers/fla/ops/kda.py
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd_kernel,
)
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_lens,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.op import exp, exp2, log
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
    check_shared_mem,
    is_intel,
    is_nvidia,
    tensor_cache,
)

if is_intel:
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )


BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

# Convert natural-log gates to log2 space before the exp2-based chunk kernels.
# log2(e) rounded to fp32, matching flash-linear-attention.
RCP_LN2 = 1.4426950216293335


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n: int) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    # ssm_state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(N, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    # if ssm_state_indices is None:
    #     stride_indices_seq, stride_indices_tok = 1, 1
    # elif ssm_state_indices.ndim == 1:
    #     stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    # else:
    #     stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        # ssm_state_indices=ssm_state_indices,
        scale=scale,
        # N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        # stride_init_state_token=stride_init_state_token,
        # stride_final_state_token=stride_final_state_token,
        # stride_indices_seq=stride_indices_seq,
        # stride_indices_tok=stride_indices_tok,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        # INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    o = o.squeeze(0)
    return o, final_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    # ssm_state_indices: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = fused_recurrent_kda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        # ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = "swish",
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.contiguous().reshape(-1, x.shape[-1])
    g = g.contiguous().reshape(-1, g.shape[-1])
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.contiguous().reshape(-1, residual.shape[-1])
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float if residual_in_fp32 else None)
    )
    y, _, _, residual_out = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        residual_dtype=residual_dtype,
        is_rms_norm=True,
    )
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))


@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    gk_scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT

    p_b = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,)
    )
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        b_kt = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        p_gk = tl.make_block_ptr(
            g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = (
            tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
            * gk_scale
        )
        # [BC, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1)) * gk_scale
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp2(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, boundary_check=(0, 1)) * gk_scale
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        # [BC, BC]
        b_ktg = b_kt * exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)

    b_A *= b_b[:, None]

    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["BK", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    gk_scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC

    p_q = tl.make_block_ptr(
        q + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)) * gk_scale

    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]

    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32) * gk_scale
        b_ktg = b_kt[None, :] * exp2(b_g - b_gk[None, :])
        b_A = tl.sum(b_k * b_ktg, 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        b_Aqk = tl.sum(b_q * b_ktg, 1)
        b_Aqk = tl.where(o_i >= j, b_Aqk * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        p_kt += H * K
        p_gk += H * K


def chunk_kda_scaled_dot_kkt_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    gk_scale: float = 1.0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor,
            in log2 space (the kernels apply `exp2`).
            Default: `None`.
        gk_scale (float):
            Scale multiplied onto `gk` as it is loaded in the kernels. Pass a natural-log
            cumsum with `gk_scale=RCP_LN2` to convert to log2 space in-kernel without
            materializing a scaled copy of `gk`. Default: `1.0`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    assert K <= 256
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BC = min(16, BT)
    NC = cdiv(BT, BC)
    BK = max(next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = (NT, NC * NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=A,
        Aqk=Aqk,
        scale=scale,
        gk_scale=gk_scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        IS_VARLEN=cu_seqlens is not None,
    )

    grid = (NT, NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=A,
        Aqk=Aqk,
        scale=scale,
        gk_scale=gk_scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        IS_VARLEN=cu_seqlens is not None,
    )
    return A, Aqk


@triton.jit(do_not_specialize=["T"])
def _recompute_w_u_fwd_kernel(
    k,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # Padded chunk_indices rows (kda_prefill_graph.py) start past T, where every
    # store below is masked; this early exit only skips their wasted work.
    if i_t * BT >= T:
        return

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

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
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


_RECOMPUTE_W_U_CONFIGS = [
    triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
    for BK in [64, 128]
    for BV in [64, 128]
    for num_warps in [2, 4, 8]
    for num_stages in [2, 3, 4]
]

recompute_w_u_fwd_kernel = triton.autotune(
    configs=_RECOMPUTE_W_U_CONFIGS,
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)(_recompute_w_u_fwd_kernel)

_K3_RECOMPUTE_W_U_CONFIGS = {
    (9, 0): {"BK": 128, "BV": 128, "num_warps": 8, "num_stages": 2},
    (10, 3): {"BK": 64, "BV": 128, "num_warps": 8, "num_stages": 2},
}


@torch.inference_mode()
def precompile_k3_recompute_w_u_kernel(
    *, num_heads: int, dtype: torch.dtype, device: torch.device
) -> bool:
    device = torch.device(device)
    if (
        not is_nvidia
        or device.type != "cuda"
        or torch.cuda.get_device_capability(device) not in _K3_RECOMPUTE_W_U_CONFIGS
    ):
        return False

    shape = (1, 1, num_heads, 128)
    k = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros_like(k)
    beta = torch.zeros((1, 1, num_heads), dtype=dtype, device=device)
    A = torch.zeros((1, 1, num_heads, 64), dtype=dtype, device=device)
    gk = torch.zeros(shape, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int64, device=device)
    recompute_w_u_fwd(k, v, beta, A, gk=gk, cu_seqlens=cu_seqlens)
    return True


def _get_k3_recompute_w_u_config(
    k: torch.Tensor,
    gk: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    K: int,
    V: int,
    BT: int,
) -> dict | None:
    if (
        not is_nvidia
        or gk is None
        or cu_seqlens is None
        or (K, V, BT) != (128, 128, 64)
    ):
        return None
    return _K3_RECOMPUTE_W_U_CONFIGS.get(torch.cuda.get_device_capability(k.device))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None
    static_config = _get_k3_recompute_w_u_config(k, gk, cu_seqlens, K, V, BT)
    kernel = (
        _recompute_w_u_fwd_kernel
        if static_config is not None
        else recompute_w_u_fwd_kernel
    )
    kernel[(NT, B * H)](
        k=k,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STORE_KG=kg is not None,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION="ieee",
        **(static_config or {}),
    )
    return w, u, kg


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # Padded chunk_indices rows (kda_prefill_graph.py) start past T, where every
    # store below is masked; this early exit only skips their wasted work.
    if i_t * BT >= T:
        return

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * V * K,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * H)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
    )
    return o


@triton.jit
def softplus_fwd(x):
    """Standard softplus: log(1 + exp(x)), with linear approx for large x."""
    return tl.where(x < 20.0, log(1.0 + exp(x)), x)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_BETA": lambda args: args["beta"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN", "HAS_BETA"],
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    beta,
    beta_out,
    beta_stride_b: tl.constexpr,
    beta_stride_t: tl.constexpr,
    beta_stride_h: tl.constexpr,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # Padded chunk_indices rows (kda_prefill_graph.py) start past T, where every
    # store below is masked; this early exit only skips their wasted work.
    if i_t * BT >= T:
        return

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(
            dt_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # Standard gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus_fwd(b_s)
    else:
        # Safe gate: lower_bound * sigmoid(exp(A_log) * (g + bias))
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Chunk-local cumulative sum
    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if HAS_BETA:
        if i_s == 0:
            offsets_t = i_t * BT + tl.arange(0, BT)
            if IS_VARLEN:
                beta_offsets = (bos + offsets_t) * beta_stride_t
            else:
                beta_offsets = i_b * beta_stride_b + offsets_t * beta_stride_t
            b_beta = tl.load(
                beta + beta_offsets + i_h * beta_stride_h,
                mask=offsets_t < T,
                other=0.0,
            ).to(tl.float32)
            # Not tl.sigmoid: tl.exp and Triton's / are approximate on CUDA. This
            # matches the unfused path's beta.float().sigmoid() bit for bit.
            tl.store(
                beta_out + (bos + offsets_t) * H + i_h,
                tl.div_rn(1.0, 1.0 + libdevice.exp(-b_beta)),
                mask=offsets_t < T,
            )


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    lower_bound: Optional[float] = None,
    beta: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Fused KDA gate activation + chunk-local cumulative sum, with optional beta.

    Combines two memory-bound kernels into one:
      1. Gate activation: g = -exp(A_log) * softplus(raw_g + dt_bias)
      2. Chunk-local cumsum along the time axis

    Args:
        g: Raw gate tensor of shape [B, T, H, K] (before activation).
        A_log: Per-head log-scale parameter, [H] elements (any shape, numel=H).
        chunk_size: Chunk size for cumsum (must be power of 2).
        scale: Optional scale factor applied to output.
        dt_bias: Optional per-head bias, flat [H*K] elements.
        cu_seqlens: Cumulative sequence lengths for variable-length input.
        output_dtype: Output dtype (default float32).
        chunk_indices: Pre-computed chunk indices for varlen mode.
        lower_bound: If set, use safe gate: lower_bound * sigmoid(exp(A_log) * g).
        beta: Optional raw beta of shape [B, T, H], including strided projections.

    Returns:
        Cumulative-summed gated tensor of shape [B, T, H, K]. If beta is
        supplied, also return its sigmoid in a contiguous float32 [B, T, H] tensor.
    """
    if cu_seqlens is not None:
        assert g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be a power of 2"
    )

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    if beta is not None:
        assert beta.shape == (B, T, H)
        assert beta.device == g.device
        beta_out = torch.empty((B, T, H), dtype=torch.float32, device=beta.device)
        beta_strides = beta.stride()
        if cu_seqlens is not None:
            beta_strides = (0, beta_strides[1], beta_strides[2])
    else:
        beta_out = None
        beta_strides = (0, 0, 0)
    if B * T == 0:
        return g if beta is None else (g, beta_out)

    def grid(meta):
        return (cdiv(meta["S"], meta["BS"]), NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        beta=beta,
        beta_out=beta_out,
        beta_stride_b=beta_strides[0],
        beta_stride_t=beta_strides[1],
        beta_stride_h=beta_strides[2],
        T=T,
        H=H,
        S=S,
        BT=BT,
    )
    return g if beta is None else (g, beta_out)


@tensor_cache
def prepare_segment_cu_seqlens(
    cu_seqlens: torch.LongTensor, segments: int, chunk_size: int
) -> Optional[torch.LongTensor]:
    """Split every sequence of ``cu_seqlens`` into ``segments`` chunk-aligned pieces.

    Returns ``[N * s + 1]`` boundaries with ``s = min(segments, min_len // chunk_size)``
    so that no segment is empty (an empty sequence would shift the sequence ids
    that ``prepare_chunk_indices`` derives from chunk-0 occurrences), or ``None``
    when nothing can be split. Segment starts are multiples of ``chunk_size``
    from their sequence start, so the global chunk enumeration (and therefore the
    per-chunk state slots and ``v_new`` rows) is identical to the unsegmented one.
    Cached per ``cu_seqlens`` object like the other varlen helpers (one host sync
    per forward).
    """
    lens = prepare_lens(cu_seqlens)
    s = min(int(segments), max(1, int(lens.min().item()) // chunk_size))
    if s <= 1:
        return None
    j = torch.arange(s, device=cu_seqlens.device, dtype=lens.dtype)
    starts = cu_seqlens[:-1, None] + (lens[:, None] * j // s) // chunk_size * chunk_size
    return torch.cat([starts.reshape(-1), cu_seqlens[-1:]]).contiguous()


@functools.lru_cache(maxsize=8)
def _eye(K: int, device: str) -> torch.Tensor:
    return torch.eye(K, device=device, dtype=torch.float32)


def _zero_identity_state(n_slots: int, H: int, V: int, K: int, device) -> torch.Tensor:
    """[n_slots, H, V + K, K] fp32: rows [0, V) zero, rows [V, V + K) identity."""
    t = torch.zeros(n_slots, H, V + K, K, device=device, dtype=torch.float32)
    t[:, :, V:, :] = _eye(K, str(device))
    return t


@tensor_cache
def prepare_segment_plan(
    cu_seqlens: torch.LongTensor,
    seg_cu_seqlens: torch.LongTensor,
    initial_state_indices: Optional[torch.Tensor],
):
    """Per-forward metadata of the segmented state pass (cached per object).

    Returns ``(slot_perm, idx0, valid_f, valid_pos, valid_all)``: the scratch
    slot of segment ``(n, s)`` in the kernel's n-major segment order (slots are
    stored s-major, ``s * N + n``, so per-segment slices are contiguous); the
    pool indices clamped at 0; a float mask of valid (``>= 0``) rows; their
    positions when some are invalid; and whether all rows are valid. The one
    host sync (``valid.all()``) happens once per forward, not per layer. Keyed
    on tensor identity only (``tensor_cache`` compares arguments with ``is``).
    """
    N = cu_seqlens.numel() - 1
    S = (seg_cu_seqlens.numel() - 1) // N
    dev = seg_cu_seqlens.device
    n = torch.arange(N, device=dev, dtype=torch.int64)
    s = torch.arange(S, device=dev, dtype=torch.int64)
    slot_perm = (s[None, :] * N + n[:, None]).reshape(-1).contiguous()
    if initial_state_indices is None:
        return slot_perm, None, None, None, True
    idx = initial_state_indices.to(torch.int64)
    valid = idx >= 0
    valid_all = bool(valid.all())
    valid_f = valid.to(torch.float32).view(N, 1, 1, 1)
    valid_pos = None if valid_all else valid.nonzero(as_tuple=True)[0]
    return slot_perm, idx.clamp_min(0), valid_f, valid_pos, valid_all


def _segmented_chunk_fwd_h(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: torch.LongTensor,
    seg_cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recurrent-state pass with the sequences split into parallel segments.

    ``chunk_gated_delta_rule_fwd_kernel_h`` is parallel only over (sequence,
    head) and walks every chunk of a sequence serially, so with few heads per
    rank it leaves most SMs idle. Per chunk the update is affine in the state
    (``h[V, K] <- h @ (diag(exp2(g_last)) - w^T k~) + u^T k~``), so a segment
    maps ``h_in -> h_in @ M_s + B_s``. Three parallel-over-segments passes:

    1. one launch with a state of width ``V + K`` whose first ``V`` rows start
       at zero with the real ``u`` (final state = ``B_s``) and whose last ``K``
       rows start at the identity with ``u = 0`` (final state = ``M_s``);
    2. a tiny sequential scan ``h_in[s+1] = h_in[s] @ M_s + B_s`` per sequence;
    3. the standard launch per segment from its true incoming state, which
       writes exactly the per-chunk ``h`` and ``v_new`` of the unsegmented run.

    The caller keeps running the output kernel with the original ``cu_seqlens``.
    The sequence final state (the last segment's slot after pass 3) is written
    back into ``initial_state`` like the in-place update of the unsegmented
    kernel; slot ``-1`` (padded rows) reads a zero state and is not written.
    The KDA block is launch-bound at this head count, so the pass is kept to a
    few launches and to no per-layer host sync.
    """
    N = cu_seqlens.numel() - 1
    S = (seg_cu_seqlens.numel() - 1) // N
    H, K = kg.shape[-2], kg.shape[-1]
    V = u.shape[-1]
    dev = kg.device
    seg_indices = prepare_chunk_indices(seg_cu_seqlens, chunk_size)
    slot_perm, idx0, valid_f, valid_pos, valid_all = prepare_segment_plan(
        cu_seqlens, seg_cu_seqlens, initial_state_indices
    )

    # Pass 1: [B_s | M_s] in one launch of state width V + K (slots s-major).
    u_ap = torch.cat([u, u.new_zeros(*u.shape[:-1], K)], dim=-1)
    state_ap = _zero_identity_state(N * S, H, V, K, dev)
    chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u_ap,
        gk=g,
        initial_state=state_ap,
        initial_state_indices=slot_perm,
        save_new_value=False,
        save_h=False,
        cu_seqlens=seg_cu_seqlens,
        chunk_indices=seg_indices,
        use_exp2=True,
    )
    del u_ap
    st = state_ap.view(S, N, H, V + K, K)
    B_s = st[:, :, :, :V, :]
    M_s = st[:, :, :, V:, :]

    # Pass 2: incoming state of every segment (fp32; right-multiplication in the
    # [V, K] layout). h_in[s] is the contiguous slot block of segment s.
    h_in = torch.empty(S, N, H, V, K, device=dev, dtype=torch.float32)
    if initial_state is None:
        h_in[0].zero_()
    else:
        h_in[0].copy_(initial_state[idx0])
        if not valid_all:
            h_in[0].mul_(valid_f)
    for s in range(S - 1):
        torch.baddbmm(
            B_s[s].reshape(N * H, V, K),
            h_in[s].view(N * H, V, K),
            M_s[s].reshape(N * H, K, K),
            out=h_in[s + 1].view(N * H, V, K),
        )
    del st, B_s, M_s, state_ap

    # Pass 3: the standard per-chunk h / v_new from the true incoming states.
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=h_in.view(S * N, H, V, K),
        initial_state_indices=slot_perm,
        save_new_value=True,
        cu_seqlens=seg_cu_seqlens,
        chunk_indices=seg_indices,
        use_exp2=True,
    )
    if initial_state is not None:
        # The kernel left each segment's exact final state in its slot; the
        # last segment's is the sequence final state.
        final = h_in[S - 1]
        if valid_all:
            initial_state.index_copy_(0, idx0, final.to(initial_state.dtype))
        elif valid_pos.numel() > 0:
            initial_state.index_copy_(
                0, idx0[valid_pos], final[valid_pos].to(initial_state.dtype)
            )
    return h, v_new


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    beta_is_raw: bool = False,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    state_segments: int = 1,
):
    chunk_size = 64
    # Pre-compute chunk indices once and thread through all downstream kernels.
    # Without this, each of the 4 callees would recompute independently. A
    # CUDA-graph caller passes padded chunk_indices / chunk_offsets it rebuilds
    # before every replay (kda_prefill_graph.py).
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if A_log is not None:
        # Fused: gate activation + chunk-local cumsum in one kernel.
        # g is raw gate (before activation); A_log, dt_bias drive the activation.
        g = kda_gate_chunk_cumsum(
            g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            beta=beta if beta_is_raw else None,
        )
        if beta_is_raw:
            g, beta = g
    else:
        # g is already gate-activated by caller; just do cumsum.
        if beta_is_raw:
            beta = beta.float().sigmoid().contiguous()
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # FUSE_DIAGONAL (fold diagonal-block compute into inter+solve) and
    # FUSE_RECOMPUTE (also fold w/u/kg recompute) save kernel launches and HBM
    # round-trips, but cost register footprint per CTA. Wins at small grid
    # where launch overhead dominates; loses at large grid where the extra
    # register pressure spills. Gate both on the same grid heuristic.
    # Total CTAs in inter_solve_fused = NT * B * H_per_rank. For varlen,
    # chunks don't cross sequence boundaries, so per-sequence ceil-divs sum to
    # more than cdiv(total_tokens, chunk_size); use chunk_indices.shape[0] which
    # already enumerates all (seq, chunk) pairs.
    _NT_pr = (
        triton.cdiv(q.shape[1], chunk_size)
        if cu_seqlens is None
        else chunk_indices.shape[0]
    )
    _H_pr = q.shape[-2]
    _B = q.shape[0]
    _small_grid = _B * _NT_pr * _H_pr <= 256
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=lower_bound is not None,
        fuse_diagonal=_small_grid,
        fuse_recompute=_small_grid,
    )

    # Segmented state pass (SGLANG_KDA_STATE_SEGMENTS): only for varlen prefill
    # without state tracking (track_chunk_idx is a per-sequence chunk index).
    seg_cu_seqlens = (
        prepare_segment_cu_seqlens(cu_seqlens, state_segments, chunk_size)
        if (
            state_segments > 1
            and cu_seqlens is not None
            and track_state is None
            # Captured prefill metadata is rebuilt in place on graph replay.
            and chunk_offsets is None
        )
        else None
    )
    if seg_cu_seqlens is not None:
        h, v_new = _segmented_chunk_fwd_h(
            kg,
            w,
            u,
            g,
            initial_state,
            initial_state_indices,
            cu_seqlens,
            seg_cu_seqlens,
            chunk_size,
        )
    else:
        h, v_new = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
            chunk_offsets=chunk_offsets,
        )
    del w, u, kg

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=v,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    del Aqk, v_new

    if output_intermediate_states:
        # h holds the recurrent state at every chunk-size boundary
        # ([1, NT, H, V, K] packed across cu_seqlens) — the mamba radix
        # track path snapshots per-chunk states from it during extend.
        return o, h
    del h
    return o


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    output_intermediate_states: bool = False,
    track_state: Optional[torch.Tensor] = None,
    track_chunk_idx: Optional[torch.Tensor] = None,
    beta_is_raw: bool = False,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    state_segments: int = 1,
    **kwargs,
):
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    # Returns o [B, T, H, V] when output_intermediate_states=False, or (o, h [B, NT, H, V, K]) when output_intermediate_states=True.
    return chunk_kda_fwd(
        q=q,
        k=k,
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta if beta_is_raw else beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        output_intermediate_states=output_intermediate_states,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
        beta_is_raw=beta_is_raw,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        state_segments=state_segments,
    )
