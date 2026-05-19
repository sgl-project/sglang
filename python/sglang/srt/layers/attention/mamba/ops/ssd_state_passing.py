# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_state_passing.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch
import triton
import triton.language as tl


@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    last_chunk_indices_ptr,
    # Matrix dimensions
    dim,
    chunk_size,
    # Strides
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_dA_cs_head,
    stride_dA_cs_chunk,
    stride_dA_cs_csize,
    stride_initstates_batch,
    stride_initstates_head,
    stride_initstates_dim,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 16,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    chunk_end = tl.load(last_chunk_indices_ptr + pid_b) + 1
    chunk_start = (
        tl.load(last_chunk_indices_ptr + pid_b - 1, mask=pid_b > 0, other=-1) + 1
    )

    states_ptr += chunk_start * stride_states_chunk + pid_h * stride_states_head
    dA_cs_ptr += (
        pid_h * stride_dA_cs_head
        + chunk_start * stride_dA_cs_chunk
        + (chunk_size - 1) * stride_dA_cs_csize
    )
    out_ptr += chunk_start * stride_out_chunk + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_INITSTATES:
        initstates_ptrs = (
            initstates_ptr
            + pid_b * stride_initstates_batch
            + pid_h * stride_initstates_head
            + offs_m * stride_initstates_dim
        )
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for _ in range(chunk_end - chunk_start):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        states = tl.exp(dA_cs) * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_cumsum,
    last_chunk_indices,
    initial_states=None,
    out_dtype=None,
):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    batch = last_chunk_indices.shape[0]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.get_device_module(states.device).device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            dA_cumsum,
            initial_states,
            last_chunk_indices,
            dim,
            chunk_size,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dA_cumsum.stride(0),
            dA_cumsum.stride(1),
            dA_cumsum.stride(2),
            initial_states_strides[0],
            initial_states_strides[1],
            initial_states_strides[2],
            HAS_INITSTATES=initial_states is not None,
        )
    return out
