"""Row-streaming KDA packed decode for gfx950.

Triton port of the CUDA kda_packed_decode kernel's layout. The generic
fused_recurrent_kda_packed_decode_kernel holds a [BV, K] fp32 state tile per
program and reaches ~5 TB/s of in-place state traffic on MI355X; this kernel
streams the state ROWS rows at a time across a per-(request, head, row-group)
program, so loads pipeline across iterations and nothing holds a full tile.
Math and op order match fused_recurrent_kda_packed_decode_kernel; outputs
match to ULPs (tl.sum reduction shape), not bits.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.op import exp
from sglang.srt.utils import is_gfx95_supported

_is_gfx95 = is_gfx95_supported()

# (max batch, rows per iteration, row groups, num_warps, num_stages), measured on
# MI355X for H=HV=16 (GLM-5.3 TP4), K=V=128, and an fp32 state pool.
_CONFIGS = (
    (12, 8, 16, 1, 3),
    (16, 8, 8, 1, 3),
    (40, 16, 8, 1, 1),
    (56, 32, 4, 2, 3),
    (None, 32, 4, 1, 1),
)


def _config(batch: int):
    for max_batch, rows, groups, warps, stages in _CONFIGS:
        if max_batch is None or batch <= max_batch:
            return rows, groups, warps, stages


@triton.jit
def kda_packed_decode_rowstream_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    lower_bound,
    o,
    state,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok,
    stride_a_tok,
    stride_b_tok,
    stride_state_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_nh, i_g = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    row0 = i_g * ROWS_PER_PROGRAM
    p_o = o + (i_n * HV + i_hv) * V + row0

    state_idx = tl.load(ssm_state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        # Padded cuda-graph slot: zero the output, leave the pool untouched.
        for r in tl.range(0, ROWS_PER_PROGRAM, ROWS):
            o_r = r + tl.arange(0, ROWS)
            tl.store(
                p_o + o_r,
                tl.zeros([ROWS], dtype=tl.float32).to(p_o.dtype.element_ty),
            )
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    x = tl.load(a + i_n * stride_a_tok + i_hv * K + o_k).to(tl.float32) + tl.load(
        dt_bias + i_hv * K + o_k
    ).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(tl.exp(A_log_val) * x)
    else:
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        b_g = -tl.exp(A_log_val) * softplus_x
    decay = exp(b_g)
    beta = tl.sigmoid(tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32))

    p_v = p_mixed + 2 * H * K + i_hv * V + row0
    p_h = state + state_idx * stride_state_slot + i_hv * V * K + row0 * K
    for r in tl.range(0, ROWS_PER_PROGRAM, ROWS):
        o_r = r + tl.arange(0, ROWS)
        p_row = p_h + o_r[:, None] * K + o_k[None, :]
        b_h = tl.load(p_row).to(tl.float32)
        b_h *= decay[None, :]
        b_v = tl.load(p_v + o_r).to(tl.float32)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o + o_r, b_o.to(p_o.dtype.element_ty))
        tl.store(p_row, b_h.to(p_row.dtype.element_ty))


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> bool:
    HV, V, K = initial_state.shape[-3:]
    return (
        _is_gfx95
        and K == 128
        and V == 128
        and a.dim() == 2
        and a.shape[1] == HV * K
        and mixed_qkv.dtype == torch.bfloat16
        and initial_state.dtype == torch.float32
        and initial_state.stride(-1) == 1
        and initial_state.stride(-2) == K
        and initial_state.stride(-3) == V * K
        and out.is_contiguous()
        and ssm_state_indices.is_contiguous()
    )


def kda_packed_decode_rowstream(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_q_heads: int,
    lower_bound: Optional[float] = None,
) -> None:
    """In-place KDA decode step with q/k l2-norm; caller must check covered()."""
    B = mixed_qkv.shape[0]
    if B == 0:
        return
    HV, V, K = initial_state.shape[-3:]
    rows, groups, num_warps, num_stages = _config(B)
    kda_packed_decode_rowstream_kernel[(B * HV, groups)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        0.0 if lower_bound is None else float(lower_bound),
        out,
        initial_state,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        initial_state.stride(0),
        H=num_q_heads,
        HV=HV,
        K=K,
        V=V,
        ROWS=rows,
        ROWS_PER_PROGRAM=V // groups,
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
