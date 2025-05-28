# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import warnings
from typing import Any, Tuple, Union

import torch
import triton
import triton.language as tl


def is_hopper_gpu():
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        return major == 9
    return False


def get_compressed_seqlens(
    cu_seqlens: torch.Tensor, kernel_size: int, kernel_stride: int
):
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.zeros(
        y_seqlens.shape[0] + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    y_cu_seqlens[1:] = torch.cumsum(y_seqlens, dim=0)
    return y_seqlens, y_cu_seqlens


def get_num_warps_stages(head_dim, block_size, is_hopper_gpu):
    """
    Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

    Args:
        head_dim (int): Size of the head dimension.
        block_size (int): Size of the block in the attention matrix.
        is_hopper_gpu (bool): True if Hopper GPU, False if Ampere GPU.

    Returns:
        tuple: (num_warps, num_stages) recommended values.
    """
    # Determine if head_dim and block_size exceed 64
    head_large = head_dim > 64
    block_large = block_size > 64

    if is_hopper_gpu:
        # Hopper GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    else:
        # Ampere GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    return num_warps, num_stages


IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    v_ptr,  # V: n x h x d
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # size and stride at compresstion
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    # skip first kernel_size query block, because they do no attend to any keys
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # attention
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # load k
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(
            off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf")
        )
        qk += tl.dot(q, k) * qk_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    # final scale
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    # save lse
    l_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh + off_q * stride_ln
    tl.store(l_ptrs, lse_i, mask=off_q < q_len)


@triton.jit
def backward_sum_o_do(
    o_ptr,  # O: n x h x d
    do_ptr,  # dO: n x h x d
    delta_ptr,  # D: h x n
    o_len,
    HEAD_DIM,
    stride_on,
    stride_oh,
    stride_od,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dh,
    stride_dn,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_n = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr
        + off_n[:, None] * stride_on
        + pid_h * stride_oh
        + off_d[None, :] * stride_od,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + off_n[:, None] * stride_don
        + pid_h * stride_doh
        + off_d[None, :] * stride_dod,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        delta_ptr + pid_h * stride_dh + off_n * stride_dn, delta, mask=off_n < o_len
    )


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DV: sh x n x kh x d
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = (
        pid_k * BLOCK_SIZE_K * kernel_stride
        + tl.arange(0, BLOCK_SIZE_K) * kernel_stride
        + kernel_size
        - 1
    )
    # load k v and keep in SRAM
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dk dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    q_lo = pid_k * BLOCK_SIZE_K * kernel_stride + kernel_size - 1
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(HEAD_DIM, q_len),
        strides=(stride_qd, stride_qn),
        offsets=(0, q_lo),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_Q),
        order=(0, 1),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(HEAD_DIM, q_len),
        strides=(stride_dod, stride_don),
        offsets=(0, q_lo),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_Q),
        order=(0, 1),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        shape=(1, q_len),
        strides=(0, stride_dn),
        offsets=(0, q_lo),
        block_shape=(1, BLOCK_SIZE_Q),
        order=(1, 0),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        shape=(1, q_len),
        strides=(0, stride_ln),
        offsets=(0, q_lo),
        block_shape=(1, BLOCK_SIZE_Q),
        order=(0, 1),
    )
    # loop for q blocks
    for i in range(q_lo, q_len, BLOCK_SIZE_Q):
        # load
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        # [BLOCK_SIZE_K, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        qk = tl.where(off_k[:, None] <= (off_q + i)[None, :], float(0.0), float("-inf"))
        qk += tl.dot(k, q) * qk_scale
        # compute p, ds
        # [BLOCK_SIZE_K, BLOCK_SIE_Q] - [1, BLOCK_SIZE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        p = tl.exp2(qk - lse)
        # [BLOCK_SIZE_K, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        dp = tl.dot(v, do)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        # update dk and dv
        # [BLOCK_SIZE_K, BLOCK_SIE_Q] @ [BLOCK_SIE_Q, HEAD_DIM] -> [BLOCK_SIZE_K, HEAD_DIM]
        dk += tl.dot(ds, tl.trans(q))
        dv += tl.dot(p, tl.trans(do))
        # increment pointers
        q_ptrs = tl.advance(q_ptrs, (0, BLOCK_SIZE_Q))
        do_ptrs = tl.advance(do_ptrs, (0, BLOCK_SIZE_Q))
        lse_ptrs = tl.advance(lse_ptrs, (0, BLOCK_SIZE_Q))
        d_ptrs = tl.advance(d_ptrs, (0, BLOCK_SIZE_Q))
    # save dk dv
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dq(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dq_ptr,
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    # skip first kernel_size query block, because they do no attend to any keys
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    # init pointers
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dq_ptrs = tl.make_block_ptr(
        base=dq_ptr + q_start * stride_dqn + pid_h * stride_dqh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_dqn, stride_dqd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_vd, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_don, stride_dod),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        shape=(q_len, 1),
        strides=(stride_dn, stride_dh),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        shape=(q_len, 1),
        strides=(stride_ln, stride_lh),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    # load q, do, lse, delta, and keep in SRAM
    q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option="zero")
    do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
    d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dq
    dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        # load
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(
            off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf")
        )
        qk += tl.dot(q, tl.trans(k)) * qk_scale
        # compute p, ds
        p = tl.exp2(qk - lse)
        dp = tl.dot(do, v)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        ds = ds.to(q.dtype)
        # update dq
        dq += tl.dot(ds, k)
        # increment pointers
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (0, BLOCK_SIZE_K))
    # save dq
    tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))


def _compressed_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert k_len == v_len and q_len > k_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.full(
        (num_q_heads, q_len),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    # launch kernel
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse


def _compressed_attention_bwd(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    sm_scale: float,
):
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    # compute D
    delta = torch.zeros([num_o_heads, o_len], device=o.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(o_len, META["BLOCK_SIZE_O"]), num_o_heads)
    BLOCK_SIZE_O = 256
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O, IS_HOPPER_GPU)
    backward_sum_o_do[grid](
        o,
        do,
        delta,
        o_len,
        head_dim,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # compute dk dv
    dk = torch.zeros(
        num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype
    )
    dv = torch.zeros(
        num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype
    )
    batch_size = cu_seqlens_q.shape[0] - 1
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)
    backward_dkdv[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dk,
        dv,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dk = dk.sum(0)
    dv = dv.sum(0)
    # compute dq
    dq = torch.zeros_like(q)
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
    )
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 64
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    backward_dq[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dq,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


class CompressedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        sm_scale=None,
    ):
        # dtype check
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _compressed_attention_fwd(
            q,
            k,
            v,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        return o, lse

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        dq, dk, dv = _compressed_attention_bwd(
            o,
            do,
            lse,
            q,
            k,
            v,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


@triton.jit
def score_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    s_ptr,
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_ln,
    stride_sh,
    stride_sq,
    stride_sk,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_bkh = tl.program_id(0)
    pid_b = pid_bkh // NUM_KV_HEADS
    pid_kh = pid_bkh % NUM_KV_HEADS
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if pid_q * BLOCK_SIZE_Q >= q_len or pid_k * BLOCK_SIZE_K >= k_len:
        return
    # init k pointer and load k
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    causal_mask = off_q[:, None] >= (off_k * kernel_stride + kernel_size - 1)[None, :]
    # init score
    s = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    # loop over gqa heads
    for h in range(NUM_SHARE_Q_HEADS):
        pid_h = pid_kh * NUM_SHARE_Q_HEADS + h
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, HEAD_DIM),
            strides=(stride_qn, stride_qd),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
            shape=(q_len, 1),
            strides=(stride_ln, stride_lh),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, 1),
            order=(0, 1),
        )
        # load q and lse
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.dot(q, k) * qk_scale
        # compute score
        s += tl.where(causal_mask, tl.exp2(qk - lse), 0)
    # save output
    s_ptrs = tl.make_block_ptr(
        base=s_ptr + pid_kh * stride_sh + q_start * stride_sq,
        shape=(q_len, k_len),
        strides=(stride_sq, stride_sk),
        offsets=(pid_q * BLOCK_SIZE_Q, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
        order=(1, 0),
    )
    tl.store(s_ptrs, s.to(s_ptr.dtype.element_ty), boundary_check=(0, 1))


def _get_attention_score(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
) -> torch.Tensor:
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert (
        lse.dtype == torch.float32
    )  # lse here is log2(sum(exp(qk*scale))), not log(sum(exp(qk*scale)))
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len > k_len
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # init score
    score = torch.zeros(
        num_k_heads, q_len, max_seqlen_k, dtype=torch.float32, device=q.device
    )
    # launch kernel
    grid = lambda META: (
        batch_size * num_k_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    score_kernel[grid](
        q,
        k,
        lse,
        score,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        lse.stride(0),
        lse.stride(1),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=8,
        num_stages=3,
    )
    return score


@triton.jit
def _transform_score_kernel(
    s_ptr,  # score, shape: [num_heads, q_len, k_len]
    bs_ptr,  # block wise score: [num_heads, q_len, num_k_block]
    offs,
    cu_seqlens_q,
    # shape
    num_heads,
    num_offs,
    max_k_len,
    max_blocks,
    pad_len,
    # kernel & block size
    block_size,
    block_stride,  # block_size // kernel_stride
    init_blocks,
    local_blocks,
    # stride
    stride_sh,
    stride_sq,
    stride_sk,
    stride_bsh,
    stride_bsq,
    stride_bsk,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = pid_k * BLOCK_SIZE_K
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return
    # load weight
    off_o = tl.arange(0, BLOCK_SIZE_O)
    w = tl.load(offs + off_o, mask=off_o < num_offs, other=0)
    # load score
    off_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    off_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) * block_stride - pad_len
    off_k = off_k[None, :] + off_o[:, None]
    s_ptrs = (
        s_ptr
        + q_start * stride_sq
        + pid_h * stride_sh
        + off_q[:, None, None] * stride_sq
        + off_k[None, :, :] * stride_sk
    )
    # weighted sum, [BQ, BO, BK] * [1, BO, 1] -> [BQ, BO, BK] -> [BQ, BK]
    s = tl.load(
        s_ptrs,
        mask=(off_q < q_len)[:, None, None] & (off_k >= 0) & (off_k < max_k_len),
        other=0,
    )
    s = s * w[None, :, None]
    s = tl.max(s, axis=1)
    # init mask and local mask
    off_bq = off_q // block_size
    off_bk = tl.arange(0, BLOCK_SIZE_K)

    s = tl.where(
        # For local blocks: set to negative infinity (exclude from topk)
        (off_bq[:, None] >= (off_bk + k_start)[None, :])
        & (off_bq[:, None] < (off_bk + k_start)[None, :] + local_blocks),
        float("-inf"),
        s,
    )

    # Keep the original conditions for init_blocks and query location as infinity
    s = tl.where(
        (off_bk[None, :] < init_blocks - k_start)
        # Force blocks where the query is located to have infinite score (always include in topk)
        | (off_bq[:, None] == (off_bk + k_start)[None, :]),
        float("inf"),
        s,
    )
    # store block wise score
    bs_ptrs = (
        bs_ptr
        + q_start * stride_bsq
        + k_start * stride_bsk
        + pid_h * stride_bsh
        + off_q[:, None] * stride_bsq
        + off_bk[None, :] * stride_bsk
    )
    tl.store(
        bs_ptrs,
        s,
        mask=(off_q < q_len)[:, None] & (off_bk < max_blocks - k_start)[None, :],
    )


def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, max_key_len = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)
    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=score.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=score.device)[None, :]
    ).view(-1)
    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())
    num_offs = int(offs.shape[0])
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE_O = triton.next_power_of_2(num_offs)
    BLOCK_SIZE_Q = 8
    grid = (
        num_k_heads * batch_size,
        triton.cdiv(total_query_len, BLOCK_SIZE_Q),
        triton.cdiv(max_blocks, BLOCK_SIZE_K),
    )
    _transform_score_kernel[grid](
        score,
        block_score,
        torch.ones_like(
            offs, dtype=offs.dtype, device=offs.device
        ),  # ! 为了max 就不用wieght了
        cu_seqlens_q,
        num_k_heads,
        offs.shape[0],
        max_key_len,
        max_blocks,
        pad_len,
        block_size,
        block_size // kernel_stride,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_score.stride(0),
        block_score.stride(1),
        block_score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        num_warps=8,
        num_stages=3,
    )
    return block_score


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    parallel_topk_compute: Union[str, bool] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        parallel_topk_compute (str, optional): Only set it to False when the sequence length is too long. This can avoid a current bug.
            We'll fix this issue later. Defaults to auto, it will be set to False when the sequence length is greater than 32k and True otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    attn_output, lse = CompressedAttention.apply(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    # do not select topk index
    if topk <= 0:
        warnings.warn("topk <= 0, returned topk_idx will be None")
        return attn_output, None

    assert topk >= init_blocks  # + local_blocks
    with torch.no_grad():
        num_k_heads, num_q_heads = k.shape[1], q.shape[1]
        num_shared_q_heads = num_q_heads // num_k_heads
        batch_size = cu_seqlens_q.shape[0] - 1
        q_idx = torch.cat(
            [
                torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                for i in range(batch_size)
            ],
            dim=0,
        )
        q_idx = q_idx // block_size
        # whether to use parallel version
        if parallel_topk_compute == "auto":
            parallel_topk_compute = cu_seqlens_q[-1] <= 32768
        # parallel version
        if parallel_topk_compute:
            # recompute score
            score = _get_attention_score(
                q,
                k,
                lse,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )
            # transform score to block-wise score
            score = transform_score(
                score,
                kernel_size,
                kernel_stride,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                init_blocks,
                local_blocks,
            )
            # get topk
            topk = min(topk, score.shape[-1])
            topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
            # print(cu_seqlens_q)
            # breakpoint()
            topk_idx[topk_idx >= q_idx[None, :, None]] = -1
            topk_idx = topk_idx.to(torch.int32)
        # non parallel version, avoid some current bugs when sequence length is too long
        # FIXME: need to fix later
        else:
            topk_idx_list = []
            for h in range(num_k_heads):
                # recompute score
                score = _get_attention_score(
                    q[:, h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    k[:, h : h + 1],
                    lse[h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    kernel_size,
                    kernel_stride,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    sm_scale,
                )
                # transform score to block-wise score
                score = transform_score(
                    score,
                    kernel_size,
                    kernel_stride,
                    block_size,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    init_blocks,
                    local_blocks,
                )
                # get topk
                topk = min(topk, score.shape[-1])
                topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
                topk_idx[topk_idx >= q_idx[None, :, None]] = -1
                topk_idx = topk_idx.to(torch.int32)
                topk_idx_list.append(topk_idx)
            topk_idx = torch.cat(topk_idx_list, dim=0)
    return attn_output, topk_idx
