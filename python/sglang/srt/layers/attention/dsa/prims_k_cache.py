"""Independent codec for the unchanged stock Prims 352-byte NVFP4 K ABI.

Bytes 0:256: adjacent low/high E2M1 nibbles for 512 NoPE features.
Bytes 256:320: 64 post-RoPE E4M3FN values. Block b's E4M3FN scale is
at 320 + 8*(b % 4) + b//4. There is NO live/global scale in this ABI.

NoPE NaN -> +0, infinities/overflow -> +/-2688; SF = RN_E4M3(amax/6).
E2M1 uses exact scaled midpoints, ties to even, not reciprocal division.
Negative zero is retained except when SF==0 (the entire block becomes +0).
Query/RoPE use SATFINITE RN_E4M3: NaN -> +0, infinities -> +/-448.
Critically, the stock loader rounds/saturates E2M1*SF to E4M3 *before*
attention; the optional bounded gather here reproduces that conversion.
All entrypoints are CUDA-only, use metadata validation (no host value reads),
and do not import or reinterpret the separate 416-byte cache codec.
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
def _decode_e2m1(c):
    m = c & 7
    a = tl.where(
        m == 0,
        0.0,
        tl.where(
            m == 1,
            0.5,
            tl.where(
                m == 2,
                1.0,
                tl.where(
                    m == 3,
                    1.5,
                    tl.where(
                        m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))
                    ),
                ),
            ),
        ),
    )
    # Bitwise sign preserves negative zero, including on zero products.
    bits = a.to(tl.uint32, bitcast=True) | ((c.to(tl.uint32) & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _write_k(
    N,
    R,
    C,
    L,
    SLOTS,
    NS: tl.constexpr,
    NF: tl.constexpr,
    RS: tl.constexpr,
    RF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    dst = tl.load(L + t).to(tl.int64)
    valid = (dst >= 0) & (dst < SLOTS)
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


def quantize_prims_k_cache_into(k_nope, k_rope, kv_buffer, loc) -> None:
    """Scatter keys without allocating expanded KV; invalid loc leaves bytes alone.

    Valid locations must be unique within a launch (duplicate writes race).
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
        n.stride(0),
        n.stride(1),
        r.stride(0),
        r.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )


@triton.jit
def _gather_k(C, I, O, SLOTS):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 1024).to(tl.int64)
    src = tl.load(I + t).to(tl.int64)
    valid = (src >= 0) & (src < SLOTS)
    row = tl.where(valid, src, 0) * 352
    # Clamp even masked addresses: never form a negative feature address.
    nd = tl.minimum(d, 511)
    packed = tl.load(C + row + nd // 2, mask=valid & (d < 512), other=0)
    c = tl.where((nd & 1) == 0, packed & 15, packed >> 4)
    b = nd // 16
    sb = tl.load(C + row + 320 + 8 * (b % 4) + b // 4, mask=valid & (d < 512), other=0)
    sf = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # This conversion is part of the STOCK LOADER, not merely debug staging.
    nope = _fp8(_decode_e2m1(c) * sf).to(tl.float32)
    rd = tl.maximum(tl.minimum(d - 512, 63), 0)
    rb = tl.load(C + row + 256 + rd, mask=valid & (d >= 512) & (d < 576), other=0)
    rope = rb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = tl.where(valid, tl.where(d < 512, nope, rope), 0.0)
    tl.store(O + t * 576 + d, value, mask=d < 576)


def dequantize_prims_k_cache_paged(kv_buffer, indices, dtype=torch.bfloat16):
    """Explicit bounded prefill/debug gather -> [indices.numel(), 1, 576].

    Never used implicitly by the writer/query path. Invalid (-1 or OOB) rows
    produce +0. Indices may be expanded/strided: contiguous() is explicit.
    """
    _cuda_same(kv_buffer, indices)
    rows = _rows(kv_buffer)
    if dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn):
        raise TypeError("gather dtype must be BF16, FP32, or E4M3FN")
    flat = _integer_flat(indices, "indices")
    out = torch.empty((flat.numel(), 1, 576), device=rows.device, dtype=dtype)
    if flat.numel():
        _gather_k[(flat.numel(),)](rows, flat, out, rows.shape[0], num_warps=4)
    return out


@triton.jit
def _cast_q(
    N,
    R,
    O,
    H,
    NT: tl.constexpr,
    NH: tl.constexpr,
    NF: tl.constexpr,
    RT: tl.constexpr,
    RH: tl.constexpr,
    RF: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 1024).to(tl.int64)
    nd = tl.minimum(d, 511)
    rd = tl.maximum(tl.minimum(d - 512, 63), 0)
    n = tl.load(N + t * NT + h * NH + nd * NF, mask=d < 512, other=0)
    r = tl.load(R + t * RT + h * RH + rd * RF, mask=(d >= 512) & (d < 576), other=0)
    q = tl.where(d < 512, n.to(tl.float32), r.to(tl.float32))
    tl.store(O + t * QT + h * QH + d * QF, _fp8(q), mask=d < 576)


def cast_prims_query_into(q_nope, q_rope, q_out) -> None:
    """Concatenate+SATFINITE RNE cast into preallocated FP8 Q; descale is 1.

    Accept output [n,H,576] or a contiguous pool view [n,1,H*576]. No
    allocations, clipping statistics, scalar reads, or Q-scale API changes.
    """
    _cuda_same(q_nope, q_rope, q_out)
    if q_nope.dtype not in _FLOAT_INPUTS or q_rope.dtype not in _FLOAT_INPUTS:
        raise TypeError("query inputs must be BF16, FP16, or FP32")
    if q_nope.ndim != 3 or q_nope.shape[2] != 512:
        raise ValueError("q_nope must be [n,H,512]")
    n, h = q_nope.shape[:2]
    if h <= 0 or q_rope.shape != (n, h, 64):
        raise ValueError("q_rope must be [n,H,64] with H > 0")
    if q_out.dtype != torch.float8_e4m3fn:
        raise TypeError("q_out must be E4M3FN")
    if q_out.shape == (n, 1, h * 576) and q_out.is_contiguous():
        out = q_out.view(n, h, 576)
    elif q_out.shape == (n, h, 576) and q_out.is_contiguous():
        out = q_out
    else:
        raise ValueError("q_out must be contiguous [n,H,576] or [n,1,H*576]")
    if n:
        _cast_q[(n, h)](
            q_nope,
            q_rope,
            out,
            h,
            *q_nope.stride(),
            *q_rope.stride(),
            *out.stride(),
            num_warps=4,
        )


__all__ = [
    "PRIMS_NVFP4_BYTES_PER_TOKEN",
    "quantize_prims_k_cache_into",
    "dequantize_prims_k_cache_paged",
    "cast_prims_query_into",
]
