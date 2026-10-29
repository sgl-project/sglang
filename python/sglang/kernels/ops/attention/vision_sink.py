"""Prefix-only correction for MiMo vision first-key logit biases."""

import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _first_key_sink_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    S,
    CU,
    qt: tl.constexpr,
    qh: tl.constexpr,
    qd: tl.constexpr,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kd: tl.constexpr,
    vt: tl.constexpr,
    vh: tl.constexpr,
    vd: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lh: tl.constexpr,
    lt: tl.constexpr,
    ss: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    GROUPS: tl.constexpr,
    LEFT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    segment, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(CU + segment), tl.load(CU + segment + 1)
    local = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    pos = start + local
    valid = pos < end
    if LEFT >= 0:
        valid = valid & (local <= LEFT)
    dims = tl.arange(0, BLOCK_D)
    kv_head = head // GROUPS
    q = tl.load(
        Q + pos[:, None] * qt + head * qh + dims[None, :] * qd,
        valid[:, None] & (dims[None, :] < D),
        0,
    ).to(tl.float32)
    k0 = tl.load(
        K + start * kt + kv_head * kh + dims * kd, (start < end) & (dims < D), 0
    ).to(tl.float32)
    logit0 = tl.sum(q * k0[None, :], axis=1) * SCALE
    lse = tl.load(LSE + head * lh + pos * lt, valid, 0)
    sink = tl.load(S + head * ss).to(tl.float32)
    # Invalid rows must be masked BEFORE exp, including padded/empty segments.
    log_p0 = tl.where(valid, logit0 - lse, -float("inf"))
    beta = libdevice.expm1(sink) * tl.exp(log_p0)
    v0 = tl.load(
        V + start * vt + kv_head * vh + dims * vd, (start < end) & (dims < DV), 0
    ).to(tl.float32)
    old = tl.load(
        OUT
        + pos[:, None] * out_stride_t
        + head * out_stride_h
        + dims[None, :] * out_stride_d,
        valid[:, None] & (dims[None, :] < DV),
        0,
    ).to(tl.float32)
    corrected = (old + beta[:, None] * v0[None, :]) / (1.0 + beta[:, None])
    tl.store(
        OUT
        + pos[:, None] * out_stride_t
        + head * out_stride_h
        + dims[None, :] * out_stride_d,
        corrected,
        valid[:, None] & (dims[None, :] < DV),
    )


def first_key_sink_inplace(q, k, v, output, lse, sinks, cu_seqlens, left, scale):
    # For full attention the launch may cover extra rows; segment bounds mask
    # them without a GPU->CPU max-length read. MiMo uses left=64 with sinks.
    prefix = q.shape[0] if left < 0 else min(q.shape[0], left + 1)
    grid = (cu_seqlens.numel() - 1, q.shape[1], triton.cdiv(prefix, 16))
    _first_key_sink_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        sinks,
        cu_seqlens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        *lse.stride(),
        sinks.stride(0),
        q.shape[-1],
        v.shape[-1],
        q.shape[1] // k.shape[1],
        left,
        scale,
        16,
        triton.next_power_of_2(max(q.shape[-1], v.shape[-1])),
        num_warps=4,
    )
