# SPDX-License-Identifier: Apache-2.0
"""Clover causal verify convolution with non-mutating state snapshots."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _clover_parallel_causal_conv(
    X,
    W,
    BIAS,
    STATE,
    SLOTS,
    INTER,
    ISLOTS,
    O,
    D: tl.constexpr,
    T: tl.constexpr,
    XS: tl.constexpr,
    XD: tl.constexpr,
    XT: tl.constexpr,
    WD: tl.constexpr,
    WW: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    SW: tl.constexpr,
    INS: tl.constexpr,
    INT: tl.constexpr,
    IND: tl.constexpr,
    INW: tl.constexpr,
    OS: tl.constexpr,
    OD: tl.constexpr,
    OUT_TOKEN_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    slot = tl.load(SLOTS + b).to(tl.int64)
    if slot < 0:
        return
    inter_slot = tl.load(ISLOTS + b).to(tl.int64)
    t = tl.arange(0, T)
    if HAS_BIAS:
        acc = tl.broadcast_to(
            tl.load(BIAS + d, d < D, other=0).to(tl.float32)[None, :], (T, BLOCK_D)
        )
    else:
        acc = tl.full((T, BLOCK_D), 0, tl.float32)
    h0 = tl.load(STATE + slot * SS + d * SD + 0 * SW, d < D, other=0)
    h1 = tl.load(STATE + slot * SS + d * SD + 1 * SW, d < D, other=0)
    h2 = tl.load(STATE + slot * SS + d * SD + 2 * SW, d < D, other=0)
    for j in tl.static_range(4):
        src_t = t - 3 + j
        xi = tl.load(
            X + b * XS + src_t[:, None] * XT + d[None, :] * XD,
            (src_t[:, None] >= 0) & (d[None, :] < D),
            other=0,
        )
        history = tl.where(
            src_t[:, None] == -3,
            h0[None, :],
            tl.where(src_t[:, None] == -2, h1[None, :], h2[None, :]),
        )
        xi = tl.where(src_t[:, None] < 0, history, xi)
        weight = tl.load(W + d * WD + j * WW, d < D, other=0)
        acc += xi * weight[None, :]
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(
        O + b * OS + t[:, None] * OUT_TOKEN_STRIDE + d[None, :] * OD,
        acc,
        d[None, :] < D,
    )
    # Deduplicated overlapping windows physically contain h1,h2,x[0:T].
    tl.store(INTER + inter_slot * INS + d * IND + 0 * INW, h1, d < D)
    tl.store(INTER + inter_slot * INS + d * IND + 1 * INW, h2, d < D)
    current = tl.load(
        X + b * XS + t[:, None] * XT + d[None, :] * XD, d[None, :] < D, other=0
    )
    tl.store(
        INTER + inter_slot * INS + (t[:, None] + 2) * INT + d[None, :] * IND,
        current,
        d[None, :] < D,
    )
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def clover_parallel_causal_conv(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    *,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    block_d=64,
):
    assert x.ndim == 3 and x.shape[2] in (16, 32) and weight.shape[1] == 4
    assert intermediate_conv_window.stride(1) == intermediate_conv_window.stride(3)
    assert activation in [None, False, True, "silu", "swish"]
    out = torch.empty_like(x)
    pdl = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _clover_parallel_causal_conv[(x.shape[0], triton.cdiv(x.shape[1], block_d))](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
        x.shape[1],
        x.shape[2],
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        *intermediate_conv_window.stride(),
        *out.stride(),
        HAS_BIAS=bias is not None,
        SILU=activation in [True, "silu", "swish"],
        BLOCK_D=block_d,
        num_warps=4,
        **pdl,
    )
    return out


def supports_clover_parallel_causal_conv(x, conv_state, weight, intermediate_windows):
    return (
        x.is_cuda
        and x.ndim == 3
        and x.shape[0] > 0
        and x.shape[2] in (16, 32)
        and x.dtype == torch.bfloat16
        and conv_state.dtype == x.dtype
        and weight.dtype == x.dtype
        and intermediate_windows.dtype == x.dtype
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (x.shape[1], 3)
        and weight.shape == (x.shape[1], 4)
        and intermediate_windows.ndim == 4
        and intermediate_windows.shape[0] >= x.shape[0]
        and intermediate_windows.shape[1:] == (x.shape[2], x.shape[1], 3)
        and intermediate_windows.stride(1) == intermediate_windows.stride(3)
    )
