"""Fused INT8 checkpoint codec for legal, unique destination slot indices.

The full five-dimensional pool strides are respected. Intermediate state and
scale math stays in FP32; scale rounding follows the original source dtype.
There are no global scratch buffers, gathers, or expanded state temporaries.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _restore(
    Q,
    S,
    D,
    SI,
    DI,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    QS3: tl.constexpr,
    QS4: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS4: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    IS: tl.constexpr,
    ID: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(2).to(tl.int64)
    n = tl.program_id(1)
    src = tl.load(SI + n * IS).to(tl.int64)
    dst = tl.load(DI + n * ID).to(tl.int64)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < H * V * K
    head = (x // (V * K)).to(tl.int64)
    v = (x // K % V).to(tl.int64)
    k = (x % K).to(tl.int64)
    q = tl.load(
        Q + layer * QS0 + src * QS1 + head * QS2 + v * QS3 + k * QS4, valid, 0
    ).to(tl.float32)
    scale = tl.load(S + layer * SS0 + src * SS1 + head * SS2 + k * SS4, valid, 0).to(
        tl.float32
    )
    value = (q * scale).to(D.dtype.element_ty)
    tl.store(D + layer * DS0 + dst * DS1 + head * DS2 + v * DS3 + k * DS4, value, valid)


@triton.jit
def _quantize(
    A,
    Q,
    S,
    SI,
    DI,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    AS3: tl.constexpr,
    AS4: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    QS3: tl.constexpr,
    QS4: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS4: tl.constexpr,
    IS: tl.constexpr,
    ID: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    k = tl.program_id(0) * BK + tl.arange(0, BK)
    v = tl.arange(0, BV)
    lh = tl.program_id(2)
    layer = (lh // H).to(tl.int64)
    head = (lh % H).to(tl.int64)
    n = tl.program_id(1)
    src = tl.load(SI + n * IS).to(tl.int64)
    dst = tl.load(DI + n * ID).to(tl.int64)
    mask = (v[:, None] < V) & (k[None, :] < K)
    a = tl.load(
        A
        + layer * AS0
        + src * AS1
        + head * AS2
        + v[:, None].to(tl.int64) * AS3
        + k[None, :].to(tl.int64) * AS4,
        mask,
        0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a), axis=0), 1.0e-8)
    # PyTorch amax propagates NaNs across the reduction column.
    has_nan = tl.sum((a != a).to(tl.int32), axis=0) > 0
    amax = tl.where(has_nan, float("nan"), amax)
    # PyTorch tensor/scalar true_divide uses an FP32 reciprocal multiplier.
    scale = (amax * (1.0 / 127.0)).to(A.dtype.element_ty).to(tl.float32)
    quotient = tl.div_rn(a, scale[None, :])
    rounded = libdevice.nearbyint(quotient)
    clipped = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    # PyTorch's NaN -> int8 conversion yields zero, including FP16 0/0.
    clipped = tl.where(quotient != quotient, 0.0, clipped)
    tl.store(
        Q
        + layer * QS0
        + dst * QS1
        + head * QS2
        + v[:, None].to(tl.int64) * QS3
        + k[None, :].to(tl.int64) * QS4,
        clipped.to(tl.int8),
        mask,
    )
    tl.store(
        S + layer * SS0 + dst * SS1 + head * SS2 + k.to(tl.int64) * SS4,
        scale.to(S.dtype.element_ty),
        k < K,
    )


def can_use(state, qdata, scale, src_slots, dst_slots):
    """Metadata-only dispatch; indices must be in range, dst slots unique."""
    return (
        torch.version.hip is None
        and state.is_cuda
        and all(t.device == state.device for t in (qdata, scale, src_slots, dst_slots))
        and state.ndim == qdata.ndim == scale.ndim == 5
        and state.shape[0] == qdata.shape[0] == scale.shape[0]
        and state.shape[2:] == qdata.shape[2:]
        and scale.shape[2:] == (state.shape[2], 1, state.shape[4])
        and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and qdata.dtype == torch.int8
        and src_slots.dtype in (torch.int32, torch.int64)
        and dst_slots.dtype in (torch.int32, torch.int64)
        and src_slots.ndim <= 1
        and dst_slots.ndim <= 1
        and src_slots.numel() == dst_slots.numel()
        and 0 < state.shape[3] <= 256
    )


def restore_to_pool(state, qdata, scale, src_slots, dst_slots):
    n = src_slots.numel()
    if n == 0:
        return
    src_slots, dst_slots = src_slots.reshape(-1), dst_slots.reshape(-1)
    l, _, h, v, k = state.shape
    ss = scale.stride()
    with torch.cuda.device(state.device):
        _restore[(triton.cdiv(h * v * k, 4096), n, l)](
            qdata,
            scale,
            state,
            src_slots,
            dst_slots,
            *qdata.stride(),
            ss[0],
            ss[1],
            ss[2],
            ss[4],
            *state.stride(),
            src_slots.stride(0),
            dst_slots.stride(0),
            h,
            v,
            k,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
        )


def quantize_from_pool(state, qdata, scale, src_slots, dst_slots):
    n = src_slots.numel()
    if n == 0:
        return
    src_slots, dst_slots = src_slots.reshape(-1), dst_slots.reshape(-1)
    l, _, h, v, k = state.shape
    ss = scale.stride()
    with torch.cuda.device(state.device):
        _quantize[(triton.cdiv(k, 32), n, l * h)](
            state,
            qdata,
            scale,
            src_slots,
            dst_slots,
            *state.stride(),
            *qdata.stride(),
            ss[0],
            ss[1],
            ss[2],
            ss[4],
            src_slots.stride(0),
            dst_slots.stride(0),
            h,
            k,
            v,
            triton.next_power_of_2(v),
            32,
            num_warps=4,
            enable_fp_fusion=False,
        )
