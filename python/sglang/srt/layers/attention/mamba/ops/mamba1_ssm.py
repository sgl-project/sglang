# Copyright 2025 SGLang Team
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
# ==============================================================================
"""Mamba1 SSM operations.

This module provides wrappers around the mamba-ssm library's selective_scan_fn
for Mamba1 models (like Jamba). The key difference from Mamba2 is the state shape:
- Mamba1: 2D temporal state (intermediate_size/tp, state_size), no groups/heads
- Mamba2: 3D temporal state (num_heads/tp, head_dim, state_size), uses n_groups
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from packaging import version

# Check for mamba-ssm library
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    selective_scan_fn = None


PAD_SLOT_ID = -1

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None}
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _mamba1_selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Matrix dimensions
    batch,
    dim,  # intermediate_size / tp
    dstate,  # state_size
    # Strides
    stride_state_batch,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_dim,
    stride_dt_bias_dim,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_dstate,
    stride_C_batch,
    stride_C_dstate,
    stride_D_dim,
    stride_z_batch,
    stride_z_dim,
    stride_out_batch,
    stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    """Triton kernel for Mamba1 selective scan update (decode step).

    Unlike Mamba2, Mamba1 doesn't have heads - it operates directly on
    the intermediate dimension.

    State shape: (batch, dim, dstate) where dim = intermediate_size / tp
    """
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr
    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch
    else:
        state_ptr += pid_b * stride_state_batch

    x_ptr += pid_b * stride_x_batch
    dt_ptr += pid_b * stride_dt_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch
    out_ptr += pid_b * stride_out_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if DT_SOFTPLUS:
        dt = softplus(dt)
    A = tl.load(
        A_ptrs,
        mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
        other=0.0,
    ).to(tl.float32)
    dA = tl.exp(A * dt[:, None])

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]

    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)

    tl.store(out_ptrs, out, mask=offs_m < dim)
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)


def mamba1_selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    out: Optional[torch.Tensor] = None,
):
    """Mamba1 selective scan update for decode (single token).

    This is the Mamba1 equivalent of selective_state_update but with 2D state
    instead of Mamba2's 3D state (no heads).

    Args:
        state: (batch, dim, dstate) - the SSM state
        x: (batch, dim) - input after conv
        dt: (batch, dim) - time step
        A: (dim, dstate) - A matrix (pre-computed as -exp(A_log))
        B: (batch, dstate) - B projection
        C: (batch, dstate) - C projection
        D: (dim,) - optional D parameter
        z: (batch, dim) - optional gate
        dt_bias: (dim,) - optional dt bias
        dt_softplus: whether to apply softplus to dt
        state_batch_indices: (batch,) - indices into state batch dim
        pad_slot_id: padding slot ID to skip
        out: (batch, dim) - optional preallocated output tensor
    """
    batch, dim, dstate = state.shape
    assert x.shape == (batch, dim)
    assert dt.shape == (batch, dim)
    assert A.shape == (dim, dstate)
    assert B.shape == (batch, dstate)
    assert C.shape == (batch, dstate)
    if D is not None:
        assert D.shape == (dim,)
    if z is not None:
        assert z.shape == (batch, dim)
    if dt_bias is not None:
        assert dt_bias.shape == (dim,)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)

    if out is None:
        out = torch.empty_like(x)
    assert out.shape == (batch, dim)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch)

    # Tune block size based on dstate
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else (
            (16, 4)
            if dstate <= 32
            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else (4, 8)))
        )
    )

    z_strides = (z.stride(0), z.stride(1)) if z is not None else (0, 0)

    with torch.cuda.device(x.device.index):
        _mamba1_selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            pad_slot_id,
            batch,
            dim,
            dstate,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            x.stride(0),
            x.stride(1),
            dt.stride(0),
            dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            D.stride(0) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            out.stride(0),
            out.stride(1),
            dt_softplus,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )

    return out


def mamba1_selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Reference implementation of Mamba1 selective scan for prefill.

    This is a pure PyTorch implementation used as fallback when mamba-ssm
    is not available. It's slower than the CUDA kernel but works on all devices.

    Args:
        u: (batch, seqlen, dim) - input
        delta: (batch, seqlen, dim) - time step
        A: (dim, dstate) - A matrix
        B: (batch, seqlen, dstate) - B projection
        C: (batch, seqlen, dstate) - C projection
        D: (dim,) - optional D parameter
        z: (batch, seqlen, dim) - optional gate
        delta_bias: (dim,) - optional delta bias
        delta_softplus: whether to apply softplus to delta
        return_last_state: whether to return the final state

    Returns:
        out: (batch, seqlen, dim) - output
        last_state: (batch, dim, dstate) - final state (if return_last_state)
    """
    batch, seqlen, dim = u.shape
    dstate = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    # Discretize
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, D, N)

    # Scan
    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=deltaA.dtype)
    ys = []
    for i in range(seqlen):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = torch.einsum("bdn,bn->bd", x, C[:, i])
        ys.append(y)

    y = torch.stack(ys, dim=1)  # (B, L, D)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)
    if z is not None:
        y = y * torch.nn.functional.silu(z)

    if return_last_state:
        return y, x
    return y


def mamba1_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Mamba1 selective scan for prefill.

    Uses the mamba-ssm CUDA kernel if available, otherwise falls back to
    the reference PyTorch implementation.

    Args:
        u: (batch, seqlen, dim) - input
        delta: (batch, seqlen, dim) - time step
        A: (dim, dstate) - A matrix
        B: (batch, seqlen, dstate) - B projection
        C: (batch, seqlen, dstate) - C projection
        D: (dim,) - optional D parameter
        z: (batch, seqlen, dim) - optional gate
        delta_bias: (dim,) - optional delta bias
        delta_softplus: whether to apply softplus to delta
        return_last_state: whether to return the final state

    Returns:
        out: (batch, seqlen, dim) - output
        last_state: (batch, dim, dstate) - final state (if return_last_state)
    """
    if HAS_MAMBA_SSM and u.is_cuda:
        # Use the optimized CUDA kernel from mamba-ssm
        # selective_scan_fn expects:
        # u: (B, D, L)
        # delta: (B, D, L)
        # A: (D, N)
        # B: (B, N, L) or (B, G, N, L)
        # C: (B, N, L) or (B, G, N, L)
        # D: (D,)
        # z: (B, D, L) optional
        # delta_bias: (D,) optional

        # Transpose inputs to match mamba-ssm expected format
        u_t = u.transpose(1, 2).contiguous()  # (B, D, L)
        delta_t = delta.transpose(1, 2).contiguous()  # (B, D, L)
        B_t = B.transpose(1, 2).contiguous()  # (B, N, L)
        C_t = C.transpose(1, 2).contiguous()  # (B, N, L)
        z_t = z.transpose(1, 2).contiguous() if z is not None else None

        out, last_state = selective_scan_fn(
            u_t,
            delta_t,
            A,
            B_t,
            C_t,
            D,
            z=z_t,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=True,
        )

        # Transpose output back
        out = out.transpose(1, 2)  # (B, L, D)

        if return_last_state:
            return out, last_state
        return out

    # Fallback to reference implementation
    return mamba1_selective_scan_ref(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )
