# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102

import triton
import triton.language as tl
from packaging import version

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_z_seqlen,
    stride_z_head,
    stride_z_hdim,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_chunk,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_start = tl.load(cu_chunk_seqlens_ptr + pid_c).to(tl.int64)
    chunk_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1).to(tl.int64)
    chunk_size_limit = chunk_end - chunk_start
    x_ptr += chunk_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += (
        chunk_start * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(
        seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1
    )

    if HAS_INITSTATES and (seq_idx != seq_idx_prev):
        prev_states_ptr = (
            initstates_ptr
            + seq_idx * stride_init_states_batch
            + pid_h * stride_init_states_head
        )
        prev_states_hdim = stride_init_states_hdim
        prev_states_dstate = stride_init_states_dstate
    else:
        prev_states_ptr = (
            states_ptr + (pid_c - 1) * stride_states_chunk + pid_h * stride_states_head
        )
        prev_states_hdim = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_k_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    C_ptrs = C_ptr + (
        offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
    )

    scale_m = tl.exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(
            C_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )

        if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
            prev_states = tl.zeros(
                (BLOCK_SIZE_DSTATE, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty
            )
        else:
            prev_states_ptrs = (
                prev_states_ptr
                + offs_n[None, :] * prev_states_hdim
                + offs_k_dstate[:, None] * prev_states_dstate
            )
            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)

        acc = tl.dot(C, prev_states) * scale_m[:, None]

    else:
        prev_states_ptrs = (
            prev_states_ptr
            + offs_n[None, :] * prev_states_hdim
            + offs_k_dstate[:, None] * prev_states_dstate
        )
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate - k),
                other=0.0,
            )
            if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
                prev_states = tl.zeros(
                    (BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty
                )
            else:
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k)
                    & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
            prev_states_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    x_ptrs = x_ptr + (
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = (
        chunk_size_limit
        if not IS_CAUSAL
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    )
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        cb *= tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        z_ptr += chunk_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]
        )
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += chunk_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


def _chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)
    assert out.shape == x.shape

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    _chunk_scan_fwd_kernel[grid](
        cb,
        x,
        z,
        out,
        dt,
        dA_cumsum,
        seq_idx,
        C,
        states,
        D,
        initial_states,
        cu_chunk_seqlens,
        chunk_size,
        headdim,
        dstate,
        seqlen,
        nheads // ngroups,
        cb.stride(0),
        cb.stride(1),
        cb.stride(2),
        cb.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dt.stride(1),
        dt.stride(0),
        dt.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        seq_idx.stride(0),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        initial_states_strides[0],
        initial_states_strides[1],
        initial_states_strides[2],
        initial_states_strides[3],
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return
