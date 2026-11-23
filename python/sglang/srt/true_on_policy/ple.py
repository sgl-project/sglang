"""Stands in for the stock PLE fusions (kernels/ops/elementwise/qwen4_gate.py, mamba/qwen4_short_conv.py): they redo
the eager BF16 chain (BF16 norms, products, gate and gated value, sign(0) = 0) and run only in decode or verify, while
prefill stays eager, so prefill and decode would differ.

Pinned Qwen4-Exp N-gram (PLE) increment shared by rollout and trainer.

One program at every token count and for prefill, chunked prefill and decode.
The gate chain matches the trainer's kernel (FP32 norms, dot, gate). The
convolution input is the gated value's grouped RMSNorm rounded to BF16, so the
serving short-conv state (model conv dtype) holds exactly what prefill consumed.
History before a segment comes from that state, or is zero when there is none;
the trainer's packed documents use zero history.

increment = BF16(gated + silu(sum_j w_j * conv_in[t - (K-1-j) * dil]))
"""

import math

import torch
import triton
import triton.language as tl

from sglang.srt.true_on_policy.hyper_connection import grouped_gemma_rmsnorm

_CONV_BLOCK_W = 256


@triton.jit(do_not_specialize=["T"])
def _ple_gate_kernel(
    key_ptr,  # BF16 [T, N*C]
    query_ptr,  # BF16 [T, N*C], branch stream
    value_ptr,  # BF16 [T, C]
    wk_ptr,
    wq_ptr,  # offset norm weights [N*C]
    gated_ptr,  # FP32 [T, N*C]
    gate_ptr,  # FP32 [T, N]
    rstdk_ptr,
    rstdq_ptr,  # FP32 [T, N]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    SQRTC: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    k = tl.load(key_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(query_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(wk_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    wq = tl.load(wq_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    rk = 1.0 / tl.sqrt(tl.sum(k * k, axis=0) / C + EPS)
    rq = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / C + EPS)
    kn = k * rk * (1.0 + wk)
    qn = q * rq * (1.0 + wq)
    score = tl.sum(kn * qn, axis=0) / SQRTC
    mag = tl.maximum(tl.abs(score), 1e-6)
    sgn = tl.where(score >= 0, 1.0, -1.0)
    gate = tl.sigmoid(sgn * tl.sqrt(mag))
    v = tl.load(value_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(gated_ptr + base + offs, gate * v, mask=mask)
    tl.store(gate_ptr + t * N + c, gate)
    tl.store(rstdk_ptr + t * N + c, rk)
    tl.store(rstdq_ptr + t * N + c, rq)


@triton.jit(do_not_specialize=["T", "W"])
def _ple_conv_kernel(
    conv_in_ptr,  # BF16 [T, W]
    gated_ptr,  # FP32 [T, W]
    convw_ptr,  # [W, K]
    offset_ptr,  # int64 [T], position of each token within its segment
    state_ptr,  # [slots, W, SL], last SL conv inputs in time order; unused without state
    slot_ptr,  # int64 [T], state slot of each token's segment
    out_ptr,  # BF16 [T, W]
    pre_ptr,  # FP32 [T, W], pre-SiLU sum
    T,
    W,
    K: tl.constexpr,
    DIL: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    wb = tl.program_id(1)
    if t >= T:
        return
    SL: tl.constexpr = (K - 1) * DIL
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < W
    o = tl.load(offset_ptr + t)
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for j in tl.static_range(K):
        lag = (K - 1 - j) * DIL
        wgt = tl.load(convw_ptr + offs * K + j, mask=mask, other=0.0).to(tl.float32)
        in_batch = o >= lag
        x = tl.load(
            conv_in_ptr + (t - lag) * W + offs, mask=mask & in_batch, other=0.0
        ).to(tl.float32)
        if HAS_STATE:
            slot = tl.load(slot_ptr + t)
            col = SL + o - lag
            hist = tl.load(
                state_ptr + slot * (W * SL) + offs * SL + col,
                mask=mask & (o < lag) & (col >= 0),
                other=0.0,
            ).to(tl.float32)
            x = tl.where(in_batch, x, hist)
        acc += wgt * x
    tl.store(pre_ptr + t * W + offs, acc, mask=mask)
    gt = tl.load(gated_ptr + t * W + offs, mask=mask, other=0.0)
    silu = acc * tl.sigmoid(acc)
    tl.store(
        out_ptr + t * W + offs, (gt + silu).to(out_ptr.dtype.element_ty), mask=mask
    )


def ple_gate(*, hc_state, key, value, norm_key_w, norm_query_w, n, eps):
    """(gated FP32 [T, W], gate, rstd_key, rstd_query FP32 [T, n])."""
    T, W = hc_state.shape
    C = W // n
    for name, x, width in (
        ("branch stream", hc_state, W),
        ("key", key, W),
        ("value", value, C),
    ):
        if x.shape != (T, width) or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError(
                f"TOP PLE requires CUDA BF16 {name} [{T}, {width}], got {tuple(x.shape)} {x.dtype}"
            )
    dev = hc_state.device
    gated = torch.empty(T, W, dtype=torch.float32, device=dev)
    gate, rstdk, rstdq = (
        torch.empty(T, n, dtype=torch.float32, device=dev) for _ in range(3)
    )
    if T:
        _ple_gate_kernel[(T * n,)](
            key.contiguous(),
            hc_state.contiguous(),
            value.contiguous(),
            norm_key_w.contiguous(),
            norm_query_w.contiguous(),
            gated,
            gate,
            rstdk,
            rstdq,
            T,
            N=n,
            C=C,
            EPS=eps,
            SQRTC=math.sqrt(C),
            BLOCK_C=triton.next_power_of_2(C),
        )
    return gated, gate, rstdk, rstdq


def ple_conv(*, conv_in, gated, conv_weight, dilation, offsets, state=None, slots=None):
    """BF16 increment and FP32 pre-SiLU sum; ``state``/``slots`` supply pre-segment history."""
    T, W = conv_in.shape
    K = conv_weight.shape[-1]
    if (
        conv_in.dtype != torch.bfloat16
        or gated.shape != (T, W)
        or gated.dtype != torch.float32
    ):
        raise ValueError("TOP PLE conv requires BF16 input and FP32 gated value")
    if offsets.shape != (T,) or offsets.dtype != torch.int64:
        raise ValueError("TOP PLE conv requires int64 per-token segment offsets")
    has_state = state is not None
    if has_state and (
        state.shape[1:] != (W, (K - 1) * dilation)
        or not state.is_contiguous()
        or slots.shape != (T,)
        or slots.dtype != torch.int64
    ):
        raise ValueError(
            "TOP PLE conv requires contiguous state [slots, W, (K-1)*dil] and int64 per-token slots"
        )
    out = torch.empty(T, W, dtype=torch.bfloat16, device=conv_in.device)
    pre = torch.empty(T, W, dtype=torch.float32, device=conv_in.device)
    if T:
        _ple_conv_kernel[(T, triton.cdiv(W, _CONV_BLOCK_W))](
            conv_in.contiguous(),
            gated,
            conv_weight.reshape(W, K).contiguous(),
            offsets,
            state if has_state else conv_in,
            slots if has_state else offsets,
            out,
            pre,
            T,
            W,
            K=K,
            DIL=dilation,
            HAS_STATE=has_state,
            BLOCK_W=_CONV_BLOCK_W,
        )
    return out, pre


def ple_increment(
    *,
    hc_state,
    key,
    value,
    norm_key_w,
    norm_query_w,
    norm_conv_w,
    conv_weight,
    n,
    eps,
    dilation,
    offsets,
    state=None,
    slots=None,
):
    """Full increment; returns (increment BF16, conv_in BF16, saved tensors for a backward)."""
    gated, gate, rstdk, rstdq = ple_gate(
        hc_state=hc_state,
        key=key,
        value=value,
        norm_key_w=norm_key_w,
        norm_query_w=norm_query_w,
        n=n,
        eps=eps,
    )
    conv_in, rstdc = grouped_gemma_rmsnorm(gated, norm_conv_w, n, eps)
    out, pre = ple_conv(
        conv_in=conv_in,
        gated=gated,
        conv_weight=conv_weight,
        dilation=dilation,
        offsets=offsets,
        state=state,
        slots=slots,
    )
    return out, conv_in, (gate, rstdk, rstdq, rstdc, pre)


def segment_offsets(cu_seqlens, total, device):
    """int64 position of every packed token within its document."""
    positions = torch.arange(total, dtype=torch.int64, device=device)
    if cu_seqlens is None:
        return positions
    cu = cu_seqlens.to(device=device, dtype=torch.int64)
    lens = (cu[1:] - cu[:-1]).clamp_min(0)
    return positions - torch.repeat_interleave(cu[:-1], lens, output_size=total)
