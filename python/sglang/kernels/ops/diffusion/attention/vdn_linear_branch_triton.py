# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for the VDN-H3 (Video DeltaNet MiniMax-H3) linear branch.

The branch is bandwidth-bound: every stage walks a ~1.4 GiB [T, H, d] bf16
tensor at the paper workload (105k rows x 56 heads x 128). Eager, each stage
is several full passes (the 5-tap temporal conv alone reads its input five
times); these kernels read each operand once and round once at the store.

    vdn_temporal_conv_act   5-tap depthwise temporal conv + SiLU [+ L2 norm]
                            (port of OpenVDN's _tconv_act_kernel)
    vdn_silu_l2norm         SiLU [+ L2 norm] over head_dim, strided input ok
    vdn_frame_stats_prep    the four GEMM operands of the frame statistics
                            (kf16, kf32, kf32 * beta, v * beta) in [F, H, S, d]
                            off one read of k and one of v -- bitwise equal
                            to the eager chain (widening casts, same products)
    vdn_linear_epilogue     RMSNorm(d) * gate with the [F, H, S, d] -> [F*S, H*d]
                            transpose folded into the store

Inference-only: they are not bitwise against the eager bf16 chains (one
rounding instead of one per op; about one bf16 ulp, closer to fp32), which is
the same contract OpenVDN documents for its inference kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_T = 16
_BLOCK_ROWS = 32


def _check_head_dim(head_dim: int) -> None:
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(
            f"head_dim must be a power of two >= 16 (tl.arange), got {head_dim}"
        )


# --------------------------------------------------------------------------
# temporal conv + SiLU + L2 norm
# --------------------------------------------------------------------------


@triton.jit
def _tconv_act_kernel(
    X,
    W,
    OUT,
    T,
    S_,
    C_,
    BLOCK_T: tl.constexpr,
    D_: tl.constexpr,
    L2: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    chan = pid_h * D_ + tl.arange(0, D_)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = rows < T

    acc = tl.zeros((BLOCK_T, D_), dtype=tl.float32)
    for dt in tl.static_range(5):
        r = rows + dt - 2
        ok = valid & (r >= 0) & (r < T)  # zero padding, both ends
        v = tl.load(
            X + (r[:, None] * S_ + pid_s) * C_ + chan[None, :],
            mask=ok[:, None],
            other=0.0,
        ).to(tl.float32)
        wd = tl.load(W + chan * 5 + dt).to(tl.float32)
        acc += v * wd[None, :]

    y = acc * tl.sigmoid(acc)  # SiLU
    if L2:
        inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
        y = y * inv[:, None]
    tl.store(
        OUT + (rows[:, None] * S_ + pid_s) * C_ + chan[None, :],
        y.to(OUT.dtype.element_ty),
        mask=valid[:, None],
    )


def vdn_temporal_conv_act(
    x: torch.Tensor,
    w: torch.Tensor,
    heads: int,
    head_dim: int,
    l2norm: bool,
) -> torch.Tensor:
    """x [T, S, C] bf16 contiguous, w [C, 5] -> [T * S, heads, head_dim]."""
    if not x.is_cuda:
        raise ValueError("vdn_temporal_conv_act is a Triton kernel; x must be on CUDA")
    _check_head_dim(head_dim)
    T, S_, C_ = x.shape
    if C_ != heads * head_dim:
        raise ValueError(f"C={C_} != heads*head_dim={heads * head_dim}")
    if w.shape != (C_, 5):
        raise ValueError(f"w must be [C, 5], got {tuple(w.shape)}")
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x)
    _tconv_act_kernel[(triton.cdiv(T, _BLOCK_T), S_, heads)](
        x,
        w,
        out,
        T,
        S_,
        C_,
        BLOCK_T=_BLOCK_T,
        D_=head_dim,
        L2=l2norm,
        num_warps=4,
        num_stages=2,
    )
    return out.view(T * S_, heads, head_dim)


# --------------------------------------------------------------------------
# SiLU + L2 norm on a (possibly strided) [N, H, d] tensor
# --------------------------------------------------------------------------


@triton.jit
def _silu_l2norm_kernel(
    X,
    OUT,
    N,
    stride_n,
    stride_h,
    H,
    BLOCK_N: tl.constexpr,
    D_: tl.constexpr,
    L2: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = rows < N
    offs = tl.arange(0, D_)
    x = tl.load(
        X + rows[:, None] * stride_n + pid_h * stride_h + offs[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    y = x * tl.sigmoid(x)
    if L2:
        inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
        y = y * inv[:, None]
    tl.store(
        OUT + (rows[:, None] * H + pid_h) * D_ + offs[None, :],
        y.to(OUT.dtype.element_ty),
        mask=valid[:, None],
    )


def vdn_silu_l2norm(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    """tokens [N, H, d] (last dim contiguous) -> contiguous [N, H, d]."""
    if not tokens.is_cuda:
        raise ValueError("vdn_silu_l2norm is a Triton kernel; tokens must be on CUDA")
    N, H, D = tokens.shape
    _check_head_dim(D)
    if tokens.stride(-1) != 1:
        tokens = tokens.contiguous()
    out = torch.empty((N, H, D), dtype=tokens.dtype, device=tokens.device)
    if N == 0:
        return out
    _silu_l2norm_kernel[(triton.cdiv(N, _BLOCK_ROWS), H)](
        tokens,
        out,
        N,
        tokens.stride(0),
        tokens.stride(1),
        H,
        BLOCK_N=_BLOCK_ROWS,
        D_=D,
        L2=l2norm,
        num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# frame statistics prologue
# --------------------------------------------------------------------------


@triton.jit
def _frame_stats_prep_kernel(
    K,
    V,
    BETA,
    K16,
    K32,
    KB32,
    VB,
    S_,
    H,
    BLOCK_S: tl.constexpr,
    D_: tl.constexpr,
):
    pid_s = tl.program_id(0)
    f = tl.program_id(1)
    h = tl.program_id(2)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = s < S_
    offs = tl.arange(0, D_)
    rows = f * S_ + s  # token rows
    src = (rows[:, None] * H + h) * D_ + offs[None, :]  # [F*S, H, d]
    dst = ((f * H + h) * S_ + s)[:, None] * D_ + offs[None, :]  # [F, H, S, d]
    k = tl.load(K + src, mask=valid[:, None], other=0.0)
    v = tl.load(V + src, mask=valid[:, None], other=0.0)
    beta = tl.load(BETA + rows * H + h, mask=valid, other=0.0)
    k32 = k.to(tl.float32)
    beta32 = beta.to(tl.float32)
    tl.store(K16 + dst, k, mask=valid[:, None])
    tl.store(K32 + dst, k32, mask=valid[:, None])
    tl.store(KB32 + dst, k32 * beta32[:, None], mask=valid[:, None])
    vb = (v.to(tl.float32) * beta32[:, None]).to(VB.dtype.element_ty)
    tl.store(VB + dst, vb, mask=valid[:, None])


def vdn_frame_stats_prep(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """key/value [F*S, H, d] bf16 contiguous, beta [F*S, H] bf16 ->
    (k16 [F,H,S,d] bf16, k32 fp32, k32*beta fp32, v*beta bf16), all contiguous."""
    if not key.is_cuda:
        raise ValueError(
            "vdn_frame_stats_prep is a Triton kernel; inputs must be on CUDA"
        )
    rows, H, D = key.shape
    _check_head_dim(D)
    if rows != num_frames * tokens_per_frame:
        raise ValueError(f"{rows} rows != {num_frames} x {tokens_per_frame}")
    key = key.contiguous()
    value = value.contiguous()
    beta = beta.to(key.dtype).contiguous()
    shape = (num_frames, H, tokens_per_frame, D)
    k16 = torch.empty(shape, dtype=key.dtype, device=key.device)
    k32 = torch.empty(shape, dtype=torch.float32, device=key.device)
    kb32 = torch.empty(shape, dtype=torch.float32, device=key.device)
    vb = torch.empty(shape, dtype=value.dtype, device=key.device)
    _frame_stats_prep_kernel[
        (triton.cdiv(tokens_per_frame, _BLOCK_ROWS), num_frames, H)
    ](
        key,
        value,
        beta,
        k16,
        k32,
        kb32,
        vb,
        tokens_per_frame,
        H,
        BLOCK_S=_BLOCK_ROWS,
        D_=D,
        num_warps=4,
    )
    return k16, k32, kb32, vb


# --------------------------------------------------------------------------
# readout epilogue: RMSNorm(d) * gate, [F, H, S, d] -> [F*S, H*d]
# --------------------------------------------------------------------------


@triton.jit
def _linear_epilogue_kernel(
    R,
    W,
    G,
    OUT,
    S_,
    H,
    eps,
    BLOCK_S: tl.constexpr,
    D_: tl.constexpr,
):
    pid_s = tl.program_id(0)
    f = tl.program_id(1)
    h = tl.program_id(2)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = s < S_
    offs = tl.arange(0, D_)
    src = ((f * H + h) * S_ + s)[:, None] * D_ + offs[None, :]
    rows = f * S_ + s
    dst = (rows[:, None] * H + h) * D_ + offs[None, :]
    r = tl.load(R + src, mask=valid[:, None], other=0.0).to(tl.float32)
    ms = tl.sum(r * r, axis=1) / D_
    w = tl.load(W + offs).to(tl.float32)
    g = tl.load(G + dst, mask=valid[:, None], other=0.0).to(tl.float32)
    y = r * (1.0 / tl.sqrt(ms + eps))[:, None] * w[None, :] * g
    tl.store(OUT + dst, y.to(OUT.dtype.element_ty), mask=valid[:, None])


def vdn_linear_epilogue(
    readout: torch.Tensor,
    norm_weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """readout [F, H, S, d] bf16 contiguous, norm_weight [d], gate [F*S, H, d]
    -> [F*S, H*d] bf16."""
    if not readout.is_cuda:
        raise ValueError(
            "vdn_linear_epilogue is a Triton kernel; readout must be on CUDA"
        )
    F, H, S_, D = readout.shape
    _check_head_dim(D)
    readout = readout.contiguous()
    gate = gate.reshape(F * S_, H, D).to(readout.dtype).contiguous()
    out = torch.empty((F * S_, H * D), dtype=readout.dtype, device=readout.device)
    _linear_epilogue_kernel[(triton.cdiv(S_, _BLOCK_ROWS), F, H)](
        readout,
        norm_weight.contiguous(),
        gate,
        out,
        S_,
        H,
        float(eps),
        BLOCK_S=_BLOCK_ROWS,
        D_=D,
        num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# boundary gather: prefix[lo-1] * prod alpha + suffix[hi+1] * prod alpha
# --------------------------------------------------------------------------


@triton.jit
def _gather_state_kernel(
    PREFIX,
    SUFFIX,
    LOGP,  # [F+1, H, dk] fp32 exclusive log-alpha prefix sums
    TEXT,  # [H, dv, dk] fp32 (or PREFIX when HAS_TEXT is False)
    BEFORE,  # [F] int32 prefix row to read (clamped)
    AFTER,  # [F] int32 suffix row to read (clamped)
    HASB,  # [F] int32 0/1
    HASA,  # [F] int32 0/1
    BRIDGEB,  # [F] int32 log-prefix row for the before side
    BRIDGEA,  # [F] int32 log-prefix row for the after side
    OUT,
    H,
    DV,
    HAS_TEXT: tl.constexpr,
    BRIDGE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    DK: tl.constexpr,
):
    f = tl.program_id(0)
    h = tl.program_id(1)
    pid_v = tl.program_id(2)
    rows = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    cols = tl.arange(0, DK)
    valid = rows < DV
    fb = tl.load(BEFORE + f)
    fa = tl.load(AFTER + f)
    has_b = tl.load(HASB + f)
    has_a = tl.load(HASA + f)
    plane = rows[:, None] * DK + cols[None, :]
    off_b = ((fb * H + h) * DV) * DK + plane
    off_a = ((fa * H + h) * DV) * DK + plane
    sb = tl.load(PREFIX + off_b, mask=valid[:, None], other=0.0)
    sa = tl.load(SUFFIX + off_a, mask=valid[:, None], other=0.0)
    if HAS_TEXT:
        ts = tl.load(TEXT + (h * DV) * DK + plane, mask=valid[:, None], other=0.0)
        sb = tl.where(has_b != 0, sb, ts)
        sa = tl.where(has_a != 0, sa, ts)
    else:
        sb = tl.where(has_b != 0, sb, 0.0)
        sa = tl.where(has_a != 0, sa, 0.0)
    if BRIDGE:
        bb = tl.load(BRIDGEB + f)
        ba = tl.load(BRIDGEA + f)
        lp_t1 = tl.load(LOGP + ((f + 1) * H + h) * DK + cols)
        lp_t = tl.load(LOGP + (f * H + h) * DK + cols)
        lp_bb = tl.load(LOGP + (bb * H + h) * DK + cols)
        lp_ba = tl.load(LOGP + (ba * H + h) * DK + cols)
        sb = sb * tl.exp(lp_t1 - lp_bb)[None, :]
        sa = sa * tl.exp(lp_ba - lp_t)[None, :]
    out = sb + sa
    tl.store(
        OUT + ((f * H + h) * DV) * DK + plane,
        out.to(OUT.dtype.element_ty),
        mask=valid[:, None],
    )


def vdn_gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    text_state: torch.Tensor | None,
    *,
    before_idx: torch.Tensor,
    after_idx: torch.Tensor,
    has_before: torch.Tensor,
    has_after: torch.Tensor,
    bridge_before: torch.Tensor,
    bridge_after: torch.Tensor,
    bridge: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """The boundary gather of the linear branch as one kernel over the
    [F, H, dv, dk] fp32 state banks (eager: ~7 passes over ~350 MB)."""
    if not prefix.is_cuda:
        raise ValueError(
            "vdn_gather_linear_state is a Triton kernel; inputs must be on CUDA"
        )
    F, H, DV, DK = prefix.shape
    _check_head_dim(DK)
    prefix = prefix.contiguous()
    suffix = suffix.contiguous()
    if bridge:
        log_alpha = torch.log(alpha.float().clamp_min(1e-12))
        logp = torch.cat(
            [torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)]
        ).contiguous()
    else:
        logp = prefix  # unused
    text = text_state.float().contiguous() if text_state is not None else prefix
    out = torch.empty(prefix.shape, dtype=out_dtype, device=prefix.device)
    i32 = lambda t: t.to(torch.int32).contiguous()  # noqa: E731
    _gather_state_kernel[(F, H, triton.cdiv(DV, _BLOCK_ROWS))](
        prefix,
        suffix,
        logp,
        text,
        i32(before_idx),
        i32(after_idx),
        i32(has_before),
        i32(has_after),
        i32(bridge_before),
        i32(bridge_after),
        out,
        H,
        DV,
        HAS_TEXT=text_state is not None,
        BRIDGE=bridge,
        BLOCK_V=_BLOCK_ROWS,
        DK=DK,
        num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# persistent frame scan: state_t = state_{t-1} @ transition_t + injection_t
# --------------------------------------------------------------------------


@triton.jit
def _scan_kernel(
    TRANS,  # [F, H, dk, dk] fp32
    INJ,  # [F, H, dv, dk] fp32
    START,  # [H, dv, dk] fp32 (or INJ when HAS_START is False)
    OUT,  # [F, H, dv, dk] fp32
    F,
    H,
    HAS_START: tl.constexpr,
    REVERSE: tl.constexpr,
    DV: tl.constexpr,
    DK: tl.constexpr,
):
    h = tl.program_id(0)
    rows = tl.arange(0, DV)
    cols = tl.arange(0, DK)
    plane = rows[:, None] * DK + cols[None, :]
    tplane = tl.arange(0, DK)[:, None] * DK + cols[None, :]
    if HAS_START:
        state = tl.load(START + (h * DV) * DK + plane)
    else:
        state = tl.zeros((DV, DK), dtype=tl.float32)
    for i in range(0, F):
        if REVERSE:
            f = F - 1 - i
        else:
            f = i
        trans = tl.load(TRANS + ((f * H + h) * DK) * DK + tplane)
        inj = tl.load(INJ + ((f * H + h) * DV) * DK + plane)
        # full fp32 (IEEE) accumulation: the recurrence runs ~100 steps and the
        # released checkpoints were trained against fp32 state math
        state = tl.dot(state, trans, inj, input_precision="ieee")
        tl.store(OUT + ((f * H + h) * DV) * DK + plane, state)


def vdn_run_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and reverse frame scans as two persistent kernels (one program
    per head and direction) instead of 2F baddbmm launches; same fp32
    recurrence, prefix[t] = frames 0..t, suffix[t] = frames t..F-1."""
    if not transitions.is_cuda:
        raise ValueError("vdn_run_scans is a Triton kernel; inputs must be on CUDA")
    F, H, DK, _ = transitions.shape
    DV = injections.shape[2]
    _check_head_dim(DK)
    _check_head_dim(DV)
    transitions = transitions.float().contiguous()
    injections = injections.float().contiguous()
    start = text_state.float().contiguous() if text_state is not None else injections
    prefix = torch.empty_like(injections)
    suffix = torch.empty_like(injections)
    common = dict(HAS_START=text_state is not None, DV=DV, DK=DK, num_warps=8)
    _scan_kernel[(H,)](
        transitions, injections, start, prefix, F, H, REVERSE=False, **common
    )
    _scan_kernel[(H,)](
        transitions, injections, start, suffix, F, H, REVERSE=True, **common
    )
    return prefix, suffix


__all__ = [
    "vdn_frame_stats_prep",
    "vdn_run_scans",
    "vdn_gather_linear_state",
    "vdn_linear_epilogue",
    "vdn_silu_l2norm",
    "vdn_temporal_conv_act",
]
