"""Reusable Triton kernels for token-probe capture and classification."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


ACT_NONE, ACT_GELU, ACT_RELU = 0, 1, 2


@triton.jit
def _classify_tail_kernel(
    H,
    W,
    B,
    Out,
    h_stride,
    o_stride,
    D: tl.constexpr,
    D_REAL: tl.constexpr,
    N: tl.constexpr,
    N_REAL: tl.constexpr,
    ACT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, N)
    d_mask = offs_d < D_REAL
    n_mask = offs_n < N_REAL
    h = tl.load(H + row * h_stride + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    # Literals, not the ACT_* names: a jit kernel cannot read module globals.
    if ACT == 1:  # ACT_GELU
        a = h * 0.5 * (1.0 + tl.erf(h * 0.7071067811865476))
    elif ACT == 2:  # ACT_RELU
        a = tl.maximum(h, 0.0)
    else:  # ACT_NONE
        a = h
    w = tl.load(
        W + offs_n[:, None] * D_REAL + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(a[None, :] * w, 1)
    if HAS_BIAS:
        acc += tl.load(B + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(Out + row * o_stride + offs_n, tl.sigmoid(acc), mask=n_mask)


def classify_tail(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: int = ACT_NONE,
) -> torch.Tensor:
    """Fused activation, classifier, bias and sigmoid for the MLP probe.

    The hidden width is a few hundred and the label count is small, so as
    separate torch ops this tail would be almost entirely launch overhead.
    """
    hidden = hidden.contiguous()
    rows, d_real = hidden.shape
    n_real = weight.shape[0]
    out = torch.empty(rows, n_real, device=hidden.device, dtype=torch.float32)
    if rows == 0:
        return out
    _classify_tail_kernel[(rows,)](
        hidden,
        weight,
        bias,
        out,
        hidden.stride(0),
        out.stride(0),
        D=triton.next_power_of_2(d_real),
        D_REAL=d_real,
        N=triton.next_power_of_2(n_real),
        N_REAL=n_real,
        ACT=act,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _add_rmsnorm_classifier_sigmoid_kernel(
    X,
    Residual,
    NormWeight,
    ClassifierWeight,
    Bias,
    Out,
    x_stride,
    residual_stride,
    out_stride,
    EPS: tl.constexpr,
    D: tl.constexpr,
    D_REAL: tl.constexpr,
    N: tl.constexpr,
    N_REAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, N)
    d_mask = offs_d < D_REAL
    n_mask = offs_n < N_REAL
    x = tl.load(X + row * x_stride + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    residual = tl.load(
        Residual + row * residual_stride + offs_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    x += residual
    x *= tl.rsqrt(tl.sum(x * x, 0) / D_REAL + EPS)
    norm_weight = tl.load(NormWeight + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    x *= norm_weight
    weight = tl.load(
        ClassifierWeight + offs_n[:, None] * D_REAL + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(x[None, :] * weight, 1)
    if HAS_BIAS:
        logits += tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(Out + row * out_stride + offs_n, tl.sigmoid(logits), mask=n_mask)


def fused_add_rmsnorm_classifier_sigmoid(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    classifier_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """FP32 add, RMSNorm, small classifier, bias and sigmoid in one launch."""
    rows, d_real = x.shape
    n_real = classifier_weight.shape[0]
    assert residual.shape == x.shape
    assert norm_weight.numel() == d_real
    assert classifier_weight.shape[1] == d_real
    if rows == 0:
        return torch.empty(0, n_real, device=x.device, dtype=torch.float32)
    if not x.is_cuda:
        summed = x.float() + residual.float()
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps)
        normalized *= norm_weight.float()
        return torch.sigmoid(
            torch.nn.functional.linear(
                normalized,
                classifier_weight.float(),
                bias.float() if bias is not None else None,
            )
        )

    out = torch.empty(rows, n_real, device=x.device, dtype=torch.float32)
    _add_rmsnorm_classifier_sigmoid_kernel[(rows,)](
        x,
        residual,
        norm_weight,
        classifier_weight,
        bias,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        EPS=eps,
        D=triton.next_power_of_2(d_real),
        D_REAL=d_real,
        N=triton.next_power_of_2(n_real),
        N_REAL=n_real,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _tap_kernel(
    H,
    R,
    Out,
    h_stride,
    h_hc_stride,
    r_stride,
    out_stride,
    col_off,
    n_cols,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.zeros([BLOCK], tl.float32)
    for hc in range(HC_MULT):
        x += tl.load(
            H + row * h_stride + hc * h_hc_stride + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * r_stride + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 0) / n_cols + eps)
    tl.store(
        Out + row * out_stride + col_off + offs,
        x.to(Out.dtype.element_ty),
        mask=mask,
    )


def tap_into(
    features: torch.Tensor,
    slot: int,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> None:
    """Write one tapped layer's input residual stream into its feature slot.

    Ordinary two-dimensional states form ``hidden_states + residual`` -- the
    value the layer's own input_layernorm is about to form. DeepSeek-V4's
    three-dimensional mHC states are summed across ``hc_mult`` instead. The
    resulting row is rms-normalized and stored in the same kernel. The store
    is also what keeps the tap alive, since the layer may overwrite its input
    further down.
    """
    assert hidden_states.ndim in (2, 3)
    if hidden_states.ndim == 3:
        assert residual is None
        n, hc_mult, h = hidden_states.shape
        h_hc_stride = hidden_states.stride(1)
    else:
        n, h = hidden_states.shape
        hc_mult = 1
        h_hc_stride = 0
    _tap_kernel[(n,)](
        hidden_states,
        residual if residual is not None else hidden_states,
        features,
        hidden_states.stride(0),
        h_hc_stride,
        residual.stride(0) if residual is not None else 0,
        features.stride(0),
        slot * h,
        h,
        eps,
        HAS_RESIDUAL=residual is not None,
        HC_MULT=hc_mult,
        BLOCK=triton.next_power_of_2(h),
        num_warps=8,
    )
