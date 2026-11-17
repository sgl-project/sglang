"""KV writer for the 352 B/token ``nvfp4.fp8rope`` DSA cache that prims_ts reads.

Row layout (FlashInfer Prims-TS ``kv_cache_format="nvfp4.fp8rope"``):
bytes 0:256 hold the 512 NoPE features as E2M1, even features in the low
nibble; bytes 256:320 hold the 64 RoPE features as E4M3; block b of 16 NoPE
features has its E4M3 scale at byte 320 + 8*(b % 4) + b//4. There is no
global scale.

NoPE NaN -> +0 and +/-inf or overflow -> +/-2688 (6 x 448); the block scale is
RN_E4M3(amax / 6) and E2M1 rounds to nearest, ties to even, against exact
scaled midpoints. A zero block scale writes +0. RoPE NaN -> +0, saturating at
+/-448. FlashInfer reads only; nothing else writes this layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PRIMS_NVFP4_BYTES_PER_TOKEN = 352
_FLOAT_INPUTS = (torch.bfloat16, torch.float16, torch.float32)


def _cuda_same(*tensors):
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("codec arguments must be tensors")
    if not all(x.is_cuda and x.device == tensors[0].device for x in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")


def _rows(cache):
    if cache.dtype != torch.uint8:
        raise TypeError("kv_buffer must be uint8")
    if cache.ndim != 3 or cache.shape[1:] != (1, 352):
        raise ValueError("kv_buffer must have shape [slots, 1, 352]")
    if not cache.is_contiguous():
        raise ValueError("kv_buffer must be contiguous")
    return cache.view(-1, 352)


def _features(x, dim, name):
    if x.dtype not in _FLOAT_INPUTS:
        raise TypeError(f"{name} must be BF16, FP16, or FP32")
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"{name} must be [n, {dim}] or [n, 1, {dim}]")
    return x  # Both token and feature strides are honored by the kernel.


def _integer_flat(x, name):
    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be int32 or int64")
    # Explicit allocation for expanded/strided metadata; callers can avoid it
    # by providing contiguous indices/locations. No hidden reshape copy.
    return x.contiguous().view(-1)


@triton.jit
def _finite(x, bound: tl.constexpr):
    return tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -bound), bound))


@triton.jit
def _fp8(x):
    return _finite(x, 448.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")


@triton.jit
def _encode_e2m1(x, sf):
    a = tl.abs(x)
    c = (
        (a > sf * 0.25).to(tl.uint8)
        + (a >= sf * 0.75).to(tl.uint8)
        + (a > sf * 1.25).to(tl.uint8)
        + (a >= sf * 1.75).to(tl.uint8)
        + (a > sf * 2.5).to(tl.uint8)
        + (a >= sf * 3.5).to(tl.uint8)
        + (a > sf * 5.0).to(tl.uint8)
    )
    sign = ((x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)) << 3
    return tl.where(sf > 0, c | sign, 0).to(tl.uint8)


@triton.jit
def _write_k(
    N,
    R,
    C,
    L,
    SLOTS,
    SKIP,
    NS: tl.constexpr,
    NF: tl.constexpr,
    RS: tl.constexpr,
    RF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    dst = tl.load(L + t).to(tl.int64)
    valid = (dst >= 0) & (dst < SLOTS) & (dst != SKIP)
    safe = tl.where(valid, dst, 0)
    b = tl.arange(0, 32).to(tl.int64)
    f = tl.arange(0, 16).to(tl.int64)
    x = tl.load(
        N + t * NS + (b[:, None] * 16 + f[None, :]) * NF, mask=valid, other=0
    ).to(tl.float32)
    x = _finite(x, 2688.0)
    sf = _fp8(tl.div_rn(tl.max(tl.abs(x), 1), 6.0))
    c = _encode_e2m1(x, sf.to(tl.float32)[:, None])
    lo, hi = tl.split(tl.reshape(c, (32, 8, 2)))
    p = tl.arange(0, 8).to(tl.int64)
    tl.store(C + safe * 352 + b[:, None] * 8 + p[None, :], lo | (hi << 4), mask=valid)
    tl.store(
        C + safe * 352 + 320 + 8 * (b % 4) + b // 4,
        sf.to(tl.uint8, bitcast=True),
        mask=valid,
    )
    r = tl.arange(0, 64).to(tl.int64)
    rope = tl.load(R + t * RS + r * RF, mask=valid, other=0).to(tl.float32)
    tl.store(
        C + safe * 352 + 256 + r, _fp8(rope).to(tl.uint8, bitcast=True), mask=valid
    )


def quantize_prims_k_cache_into(
    k_nope, k_rope, kv_buffer, loc, *, reserved_skip_index: int = 0
) -> None:
    """Scatter keys without allocating expanded KV; invalid loc leaves bytes alone.

    Like the other MLA writers, writes to ``reserved_skip_index`` are skipped:
    slot 0 is where padding tokens point (pass -1 to write it). Other valid
    locations must be unique within a launch (duplicate writes race).
    BF16 is the model input; FP16/FP32 inputs are also accepted explicitly.
    """
    _cuda_same(k_nope, k_rope, kv_buffer, loc)
    n = _features(k_nope, 512, "k_nope")
    r = _features(k_rope, 64, "k_rope")
    rows = _rows(kv_buffer)
    locations = _integer_flat(loc, "loc")
    if n.shape[0] != r.shape[0] or n.shape[0] != locations.numel():
        raise ValueError("key and loc token counts must match")
    if not n.shape[0]:
        return
    _write_k[(n.shape[0],)](
        n,
        r,
        rows,
        locations,
        rows.shape[0],
        reserved_skip_index,
        n.stride(0),
        n.stride(1),
        r.stride(0),
        r.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )


__all__ = ["PRIMS_NVFP4_BYTES_PER_TOKEN", "quantize_prims_k_cache_into"]
