# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102

import torch
import triton
import triton.language as tl
from packaging import version

from sglang.srt.utils import is_npu

_is_npu = is_npu()
TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    out_x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    chunk_indices_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_z_batch,
    stride_z_seqlen,
    stride_z_head,
    stride_z_hdim,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_batch,
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
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    if not HAS_INITSTATES:
        c_idx = pid_c
        c_off = 0
    else:
        c_idx = tl.load(chunk_indices_ptr + pid_c, mask=pid_c > -1, other=0)
        c_off = tl.load(chunk_offsets_ptr + pid_c, mask=pid_c > -1, other=0)

    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += (
        pid_b * stride_cb_batch
        + c_idx * stride_cb_chunk
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + c_idx * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + c_idx * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + c_idx * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    C_ptr += (
        pid_b * stride_C_batch
        + c_idx * chunk_size * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
    prev_states_ptr = (
        states_ptr
        + pid_b * stride_states_batch
        + c_idx * stride_states_chunk
        + pid_h * stride_states_head
    )
    prev_states_hdim = stride_states_hdim
    prev_states_dstate = stride_states_dstate

    chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + c_idx * chunk_size * stride_seq_idx_seqlen
        )

        # - we only need seq_idx_prev to be aligned to chunk boundary
        seq_idx_prev = tl.load(
            seq_idx_ptr - stride_seq_idx_seqlen, mask=c_idx >= 1, other=0
        )

        if HAS_INITSTATES:
            # if there are init states, we only need seq_idx_m to point
            # what is the current seq_idx

            # get current seq idx
            if (pid_m * BLOCK_SIZE_M + c_off) < chunk_size_limit:
                seq_idx_m = tl.load(
                    seq_idx_ptr
                    + (pid_m * BLOCK_SIZE_M + c_off) * stride_seq_idx_seqlen,
                )

                # - recall that in ssd_state_passing, for the case c_off == 0
                # i.e., the very first sequence, we made states_ptr hold its initial state
                # so this edge case is taken care of
                if (
                    (c_off == 0)
                    and (
                        seq_idx_prev != seq_idx_m
                    )  # if a seq is changed exactly on boundary
                    or (c_off > 0)  # implies a new example (pseudo chunk)
                ):

                    # - replace prev_states_ptr with init_states
                    prev_states_ptr = (
                        initstates_ptr
                        + seq_idx_m * stride_init_states_batch
                        + pid_h * stride_init_states_head
                    )
                    prev_states_hdim = stride_init_states_hdim  # override strides
                    prev_states_dstate = stride_init_states_dstate

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    # - handle chunk state limit
    if HAS_INITSTATES:

        # have to split this if otherwise compilation will have problems
        dA_cs_m_boundary = 0.0

        # get the c_idx for the next (logica) chunk
        c_idx_n = tl.load(
            chunk_indices_ptr + (pid_c + 1),
            mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
            other=-1,  # to trigger different chunk
        )

        # - there are things to consider
        # A. if c_off > 0 then we need to move the dA_cs boundary to ensure correct
        #    contribution of past states
        # B. if c_off_n < chunk_size_limit, then we need to adjust this so as not to
        #    encroach into the next sequence, where c_off_n is the offset of the next
        #    (logical) chunk.
        # An equivalent check for B is c_idx == c_idx_n, where there is repetition in
        # (logical) chunk indices.

        if (c_idx == c_idx_n) or c_off > 0:

            # get the next offset
            c_off_n = tl.load(
                chunk_offsets_ptr + (pid_c + 1),
                mask=pid_c > -1 and (pid_c + 1) < chunk_meta_num,
                other=chunk_size,
            )

            # in this case, adjust down the chunk_size_limit
            if c_idx == c_idx_n:
                chunk_size_limit = min(c_off_n, chunk_size_limit)

            # get the cs at the offset boundary
            # - c_off == 0 is a passthrough
            # - We need dA_cs at the boundary, defined by c_off - no need
            #   to increase pointer by pid_m (it is a constant offset,
            #   i.e. the same for all blocks)
            dA_cs_m_boundary = tl.load(
                dA_cumsum_ptr + (c_off - 1) * stride_dA_cs_csize,
                mask=(((c_off - 1) > -1) and ((c_off) < chunk_size)),
                other=0.0,
            ).to(tl.float32)

    if HAS_SEQ_IDX:
        # - handle seq idx when HAS_INITSTATES==False
        if not HAS_INITSTATES:
            seq_idx_m = tl.load(
                seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                mask=offs_m < chunk_size_limit,
                other=-1,
            )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or c_idx > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(
            0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
        )
        C_ptrs = C_ptr + (
            offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
        )

        prev_states_ptrs = prev_states_ptr + (
            offs_n[None, :] * prev_states_hdim
            + offs_k_dstate[:, None] * prev_states_dstate
        )
        if HAS_SEQ_IDX:

            if not HAS_INITSTATES:
                # - this is for continuous batching where there is no init states
                scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
            else:
                # - if there is initstates, we will rely on prev_states, no zeroing
                #   required.
                scale_m = tl.exp(dA_cs_m - dA_cs_m_boundary)
        else:
            scale_m = tl.exp(dA_cs_m)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate),
                other=0.0,
            )

            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(
                    C_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit)
                    & (offs_k_dstate[None, :] < dstate - k),
                    other=0.0,
                )
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
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

    offs_k = tl.arange(0, BLOCK_SIZE_K) + c_off
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
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
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

    offs_out_m = pid_m * BLOCK_SIZE_M + c_off + tl.arange(0, BLOCK_SIZE_M)
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
        out_x_ptr += (
            pid_b * stride_out_batch
            + c_idx * chunk_size * stride_out_seqlen
            + pid_h * stride_out_head
        )
        out_x_ptrs = out_x_ptr + (
            stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :]
        )
        tl.store(
            out_x_ptrs,
            acc,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
        )

        z_ptr += (
            pid_b * stride_z_batch
            + c_idx * chunk_size * stride_z_seqlen
            + pid_h * stride_z_head
        )
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

    out_ptr += (
        pid_b * stride_out_batch
        + c_idx * chunk_size * stride_out_seqlen
        + pid_h * stride_out_head
    )
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
    D=None,
    z=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    initial_states=None,
    out=None,
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)

    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

        if initial_states is not None:
            # with initial states, we need to take care of how
            # seq_idx crosses the boundaries
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert (
                chunk_indices is not None and chunk_offsets is not None
            ), "chunk_indices and chunk_offsets should have been set"
        else:
            chunk_indices, chunk_offsets = None, None
    else:
        chunk_indices, chunk_offsets = None, None

    assert out.shape == x.shape

    if z is not None:
        out_x = torch.empty_like(x)
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks if chunk_offsets is None else len(chunk_offsets),
        nheads,
    )
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3))
        if z is not None
        else (0, 0, 0, 0)
    )
    _chunk_scan_fwd_kernel[grid](
        cb,
        x,
        z,
        out,
        out_x,
        dt,
        dA_cumsum,
        seq_idx,
        C,
        states,
        D,
        initial_states,
        chunk_indices,
        chunk_offsets,
        len(chunk_indices) if chunk_indices is not None else 0,
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,
        cb.stride(0),
        cb.stride(1),
        cb.stride(2),
        cb.stride(3),
        cb.stride(4),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        z_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        *(
            (
                initial_states.stride(0),
                initial_states.stride(1),
                initial_states.stride(2),
                initial_states.stride(3),
            )
            if initial_states is not None
            else (0, 0, 0, 0)
        ),
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return out_x


def _chunk_scan_fwd_native(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    D=None,
    z=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    initial_states=None,
    out=None,
):
    """
    PyTorch native implementation of chunk scan forward pass.
    This avoids Triton compilation issues on Ascend NPU.
    
    Handles pseudo-chunk mechanism for continuous batching:
    When chunk_indices/chunk_offsets are provided, iterates over logical chunks
    (which handle sequence boundaries within physical chunks) instead of
    physical chunks directly.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    nheads_ngroups_ratio = nheads // ngroups

    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert out.shape == x.shape

    # Handle seq_idx and initial_states
    has_initial_states = False
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
        if initial_states is not None:
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert chunk_indices is not None and chunk_offsets is not None
            has_initial_states = True
        else:
            chunk_indices, chunk_offsets = None, None
    else:
        chunk_indices, chunk_offsets = None, None

    # Prepare output
    if z is not None:
        out_x = torch.empty_like(x)
    else:
        out_x = None

    def _process_logical_chunk(b, c_idx, c_off, chunk_size_limit, prev_states_chunk):
        """Process a single logical chunk. All intermediates computed in float32 to match Triton."""
        n_pos = chunk_size_limit - c_off
        if n_pos <= 0:
            return

        start_pos = c_idx * chunk_size + c_off
        end_pos = start_pos + n_pos

        # Get chunk data — convert to float32 for computation (Triton uses float32 accumulators)
        x_chunk = x[b, start_pos:end_pos, :, :].float()  # (n_pos, nheads, headdim)
        C_chunk = C[b, start_pos:end_pos, :, :].float()  # (n_pos, ngroups, dstate)
        dt_chunk = dt[b, :, c_idx, c_off:c_off + n_pos].float()  # (nheads, n_pos)
        dA_cs_chunk = dA_cumsum[b, :, c_idx, c_off:c_off + n_pos]  # (nheads, n_pos) already float32
        cb_chunk = cb[b, c_idx, :, c_off:c_off + n_pos, c_off:c_off + n_pos].float()  # (ngroups, n_pos, n_pos)

        # Expand C for all heads: (n_pos, ngroups, dstate) -> (n_pos, nheads, dstate)
        C_expanded = C_chunk.unsqueeze(2).expand(-1, -1, nheads_ngroups_ratio, -1)
        C_expanded = C_expanded.reshape(n_pos, nheads, dstate)

        # Expand cb for all heads: (ngroups, n_pos, n_pos) -> (nheads, n_pos, n_pos)
        cb_expanded = cb_chunk.unsqueeze(1).expand(-1, nheads_ngroups_ratio, -1, -1)
        cb_expanded = cb_expanded.reshape(nheads, n_pos, n_pos)

        # Compute dA_cs_boundary for offset correction
        if c_off > 0:
            dA_cs_boundary = dA_cumsum[b, :, c_idx, c_off - 1]  # (nheads,) float32
        else:
            dA_cs_boundary = torch.zeros(nheads, device=x.device, dtype=torch.float32)

        # Compute scale_m for C @ states contribution
        if has_initial_states:
            # With initial states: scale = exp(dA_cs - boundary)
            scale = torch.exp(dA_cs_chunk - dA_cs_boundary.unsqueeze(-1))  # (nheads, n_pos)
        elif seq_idx is not None:
            # Without initial states but with seq_idx:
            # Zero out contribution if sequence changed at chunk boundary
            if c_idx > 0:
                seq_idx_prev = seq_idx[b, c_idx * chunk_size - 1]
            else:
                seq_idx_prev = 0
            seq_idx_m = seq_idx[b, start_pos:end_pos]  # (n_pos,)
            # scale = where(seq_idx_m == seq_idx_prev, exp(dA_cs), 0)
            seq_match = (seq_idx_m == seq_idx_prev).unsqueeze(0)  # (1, n_pos)
            scale = torch.where(seq_match, torch.exp(dA_cs_chunk), torch.zeros_like(dA_cs_chunk))
        else:
            scale = torch.exp(dA_cs_chunk)  # (nheads, n_pos)

        # Part 1: C @ prev_states * scale (all in float32)
        # prev_states_chunk: (nheads, headdim, dstate)
        C_t = C_expanded.transpose(0, 1)  # (nheads, n_pos, dstate) float32
        states_t = prev_states_chunk.float().transpose(1, 2)  # (nheads, dstate, headdim) float32
        C_states = torch.bmm(C_t, states_t)  # (nheads, n_pos, headdim) float32
        C_states = C_states * scale.unsqueeze(-1)  # apply scale
        C_states = C_states.transpose(0, 1)  # (n_pos, nheads, headdim)

        # Part 2: causal cb @ x with dt and dA_cs weighting
        # IMPORTANT: Must apply causal mask BEFORE exp to avoid overflow.
        # In the upper triangle (m < k), dA_cs_m - dA_cs_k is large positive,
        # which causes exp() to overflow to inf. Then inf * 0 (causal mask) = NaN.
        # The Triton kernel avoids this by never computing the upper triangle at all.
        causal_mask = torch.tril(torch.ones(n_pos, n_pos, device=x.device, dtype=torch.bool)).unsqueeze(0)  # (1, n_pos, n_pos)
        dA_cs_m = dA_cs_chunk.unsqueeze(-1)  # (nheads, n_pos, 1)
        dA_cs_k = dA_cs_chunk.unsqueeze(1)   # (nheads, 1, n_pos)
        dt_expanded = dt_chunk.unsqueeze(1)   # (nheads, 1, n_pos) - dt at K positions, broadcast over M
        dA_diff = dA_cs_m - dA_cs_k  # (nheads, n_pos, n_pos)
        # Zero upper triangle before exp to prevent overflow (exp(0)=1, safe)
        dA_diff = dA_diff.masked_fill(~causal_mask, 0.0)
        cb_scaled = cb_expanded * torch.exp(dA_diff) * dt_expanded
        # Zero upper triangle of result (where exp gave 1 instead of needed 0)
        cb_scaled = cb_scaled.masked_fill(~causal_mask, 0.0)

        # cb @ x: (nheads, n_pos, n_pos) @ (nheads, n_pos, headdim) — all float32
        x_chunk_t = x_chunk.transpose(0, 1)  # (nheads, n_pos, headdim) float32
        cb_x = torch.bmm(cb_scaled, x_chunk_t)  # (nheads, n_pos, headdim) float32
        cb_x = cb_x.transpose(0, 1)  # (n_pos, nheads, headdim)

        # Combine
        out_chunk = C_states + cb_x

        # Add D term (convert D to float32 to match out_chunk on NPU)
        if D is not None:
            if D.dim() == 2:
                D_exp = D.float().unsqueeze(0)  # (1, nheads, headdim)
            else:
                D_exp = D.float().unsqueeze(0).unsqueeze(-1)  # (1, nheads, 1)
            out_chunk = out_chunk + x_chunk * D_exp

        # Apply SiLU gating (convert z to float32 to avoid mixed dtype on NPU)
        if z is not None:
            z_chunk = z[b, start_pos:end_pos, :, :].float()
            if out_x is not None:
                out_x[b, start_pos:end_pos, :, :] = out_chunk
            z_silu = z_chunk * torch.sigmoid(z_chunk)
            out_chunk = out_chunk * z_silu

        out[b, start_pos:end_pos, :, :] = out_chunk

    # Main loop
    if not has_initial_states:
        # Standard path: iterate over physical chunks
        for b in range(batch):
            for c in range(nchunks):
                chunk_size_limit = min(chunk_size, seqlen - c * chunk_size)
                if chunk_size_limit <= 0:
                    continue
                prev_states = states[b, c, :, :, :]  # (nheads, headdim, dstate)
                _process_logical_chunk(b, c, 0, chunk_size_limit, prev_states)
    else:
        # Pseudo-chunk path: iterate over logical chunks
        num_logical_chunks = len(chunk_indices)
        for b in range(batch):
            for lc in range(num_logical_chunks):
                c_idx = chunk_indices[lc].item()
                c_off = chunk_offsets[lc].item()

                # Determine chunk_size_limit
                chunk_size_limit = min(chunk_size, seqlen - c_idx * chunk_size)

                # If next logical chunk is in same physical chunk, limit to its offset
                if lc + 1 < num_logical_chunks:
                    c_idx_next = chunk_indices[lc + 1].item()
                    if c_idx == c_idx_next:
                        c_off_next = chunk_offsets[lc + 1].item()
                        chunk_size_limit = min(c_off_next, chunk_size_limit)

                if chunk_size_limit - c_off <= 0:
                    continue

                # Determine prev_states: use states or initial_states
                start_pos = c_idx * chunk_size + c_off
                prev_states = states[b, c_idx, :, :, :]  # default

                # Check if we need to use initial_states at sequence boundary
                if c_idx > 0:
                    seq_idx_prev = seq_idx[b, c_idx * chunk_size - 1].item()
                else:
                    seq_idx_prev = 0

                if start_pos < seqlen:
                    seq_idx_m = seq_idx[b, start_pos].item()
                else:
                    seq_idx_m = seq_idx_prev

                # Replace with initial_states if at sequence boundary
                if (c_off == 0 and seq_idx_prev != seq_idx_m) or c_off > 0:
                    prev_states = initial_states[seq_idx_m, :, :, :]  # (nheads, headdim, dstate)

                _process_logical_chunk(b, c_idx, c_off, chunk_size_limit, prev_states)

    return out_x


if _is_npu:
    _chunk_scan_fwd = _chunk_scan_fwd_native
