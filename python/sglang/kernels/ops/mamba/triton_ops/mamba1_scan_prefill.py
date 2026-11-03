# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's XPU `_selective_scan_fwd_kernel` (vllm/_xpu_ops.py,
# vllm-project/vllm#43421). Changes: reads SGLang's (tokens, dim) tensors through
# strides, one program per (sequence, BLOCK_D channels) so per-token loads of
# x/dt/z are contiguous, no prefix-cache (APC) path, separate output buffer,
# dt bias already applied by dt_proj.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _mamba1_scan_prefill_kernel(
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_ptr,
    idx_ptr,
    qsl_ptr,
    init_ptr,
    dim,
    dstate,
    sx_t,
    sx_d,
    sdt_t,
    sdt_d,
    sA_d,
    sA_n,
    sB_t,
    sB_n,
    sC_t,
    sC_n,
    sz_t,
    sz_d,
    so_t,
    so_d,
    ss_b,
    ss_d,
    ss_n,
    HAS_INIT: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq = tl.program_id(0)
    pd = tl.program_id(1)

    start = tl.load(qsl_ptr + seq).to(tl.int64)
    end = tl.load(qsl_ptr + seq + 1).to(tl.int64)
    slot = tl.load(idx_ptr + seq).to(tl.int64)

    offs_d = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    md = offs_d < dim
    mn = offs_n < dstate
    m2 = md[:, None] & mn[None, :]

    A = tl.load(
        A_ptr + offs_d[:, None] * sA_d + offs_n[None, :] * sA_n, mask=m2, other=0.0
    ).to(tl.float32)
    Dv = tl.load(D_ptr + offs_d, mask=md, other=0.0).to(tl.float32)

    s_ptrs = state_ptr + slot * ss_b + offs_d[:, None] * ss_d + offs_n[None, :] * ss_n
    if HAS_INIT:
        flag = tl.load(init_ptr + seq)
        h = tl.load(s_ptrs, mask=m2 & (flag != 0), other=0.0).to(tl.float32)
    else:
        h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    for t in range(start, end):
        x = tl.load(x_ptr + t * sx_t + offs_d * sx_d, mask=md, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + t * sdt_t + offs_d * sdt_d, mask=md, other=0.0).to(
            tl.float32
        )
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
        z = tl.load(z_ptr + t * sz_t + offs_d * sz_d, mask=md, other=0.0).to(tl.float32)
        Bv = tl.load(B_ptr + t * sB_t + offs_n * sB_n, mask=mn, other=0.0).to(
            tl.float32
        )
        Cv = tl.load(C_ptr + t * sC_t + offs_n * sC_n, mask=mn, other=0.0).to(
            tl.float32
        )

        dA = tl.exp(dt[:, None] * A)
        h = dA * h + (dt * x)[:, None] * Bv[None, :]
        y = tl.sum(h * Cv[None, :], axis=1) + Dv * x
        y = y * (z / (1.0 + tl.exp(-z)))
        tl.store(
            out_ptr + t * so_t + offs_d * so_d, y.to(out_ptr.dtype.element_ty), mask=md
        )

    tl.store(s_ptrs, h.to(state_ptr.dtype.element_ty), mask=m2)


def mamba1_scan_prefill(
    *,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    block_d: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Mamba-1 selective scan over a varlen prefill batch, in one launch.

    x, dt, gate, out: (T, dim), any strides; dt is pre-softplus with bias applied.
    A: (dim, dstate) fp32.  B, C: (T, dstate).  D: (dim,).
    ssm_state: (slots, dim, 1, dstate) or (slots, dim, dstate). Sequence i starts
    from ssm_state[state_indices[i]] when has_initial[i] (else zeros), and its
    final state is written back to that slot.
    query_start_loc: (num_seqs + 1,) int.  state_indices: (num_seqs,) int.
    Returns out, holding y = (C . h + D * x) * silu(gate) per token.
    """
    num_seqs = query_start_loc.shape[0] - 1
    dim, dstate = A.shape
    st = ssm_state.squeeze(2) if ssm_state.dim() == 4 else ssm_state
    assert st.shape[1:] == (dim, dstate), (st.shape, A.shape)
    assert x.shape == dt.shape == gate.shape == out.shape, "x/dt/gate/out shape"
    assert state_indices.shape[0] == num_seqs
    if num_seqs == 0:
        return out
    grid = (num_seqs, triton.cdiv(dim, block_d))
    has_init = has_initial is not None
    _mamba1_scan_prefill_kernel[grid](
        x,
        dt,
        A,
        B,
        C,
        D,
        gate,
        out,
        st,
        state_indices,
        query_start_loc,
        # Unused when HAS_INIT is False; any valid pointer satisfies the launch.
        has_initial if has_init else query_start_loc,
        dim,
        dstate,
        x.stride(0),
        x.stride(1),
        dt.stride(0),
        dt.stride(1),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        gate.stride(0),
        gate.stride(1),
        out.stride(0),
        out.stride(1),
        st.stride(0),
        st.stride(1),
        st.stride(2),
        HAS_INIT=has_init,
        DT_SOFTPLUS=dt_softplus,
        BLOCK_D=block_d,
        BLOCK_N=triton.next_power_of_2(dstate),
        num_warps=num_warps,
    )
    return out
