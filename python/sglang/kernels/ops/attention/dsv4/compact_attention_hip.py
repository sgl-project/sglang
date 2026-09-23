"""Sparse attention reading V4.1 FP8/FP4 pages directly on HIP."""

import torch
import triton
from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import _qkpv_fp8
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
    aiter_sparse_split_reduce,
)

RCP_LN2: gl.constexpr = 1.4426950408889634  # exp(x) = exp2(x * RCP_LN2)


@gluon.jit
def _load_compact(Cache, offset, valid, USE_BUFFER: gl.constexpr):
    if USE_BUFFER:
        return gl.amd.cdna4.buffer_load(Cache, offset.to(gl.int32), valid, other=0)
    return gl.load(Cache + offset, valid, 0)


@gluon.jit
def _read_compact_gluon(
    Cache,
    slots,
    valid,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    FP4: gl.constexpr,
    USE_BUFFER: gl.constexpr,
    G: gl.constexpr,
):
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, G))
    base = (slots // PAGE).to(gl.int64) * STRIDE
    row = slots % PAGE
    if FP4:
        PG: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
        pd = gl.arange(0, 256, layout=gl.SliceLayout(0, PG))
        packed_base = gl.convert_layout(base, gl.SliceLayout(1, PG))
        packed_row = gl.convert_layout(row, gl.SliceLayout(1, PG))
        packed_valid = gl.convert_layout(valid, gl.SliceLayout(1, PG))
        packed = _load_compact(
            Cache,
            packed_base[:, None] + packed_row[:, None] * 256 + pd[None, :],
            packed_valid[:, None],
            USE_BUFFER,
        )
        value = gl.amd.cdna4.scaled_upcast(
            packed,
            gl.full((slots.shape[0], 512), 127, gl.uint8, G),
            gl.bfloat16,
            axis=1,
        ).to(gl.float32)
        scale = (
            _load_compact(
                Cache,
                base[:, None] + PAGE * 256 + row[:, None] * 32 + d[None, :] // 16,
                valid[:, None],
                USE_BUFFER,
            )
            .to(gl.float8e4nv, bitcast=True)
            .to(gl.float32)
        )
    else:
        value = (
            _load_compact(
                Cache,
                base[:, None] + row[:, None] * 512 + d[None, :],
                valid[:, None],
                USE_BUFFER,
            )
            .to(gl.float8e4nv, bitcast=True)
            .to(gl.float32)
        )
        exp = _load_compact(
            Cache,
            base[:, None] + PAGE * 512 + row[:, None] * 16 + d[None, :] // 32,
            valid[:, None],
            USE_BUFFER,
        ).to(gl.int32)
        scale = gl.where(exp == 0, 0x00400000, exp << 23).to(gl.float32, bitcast=True)
        scale = gl.where(exp == 255, float("nan"), scale)
    return (value * scale).to(gl.bfloat16)


@gluon.jit
def _compact_qkpv(
    kv,
    valid,
    qdot,
    m,
    den,
    acc,
    head_mask,
    smem,
    MF: gl.constexpr,
    H: gl.constexpr,
    BLOCK: gl.constexpr,
    SCALE: gl.constexpr,
):
    return _qkpv_fp8(
        kv,
        kv,
        valid,
        qdot,
        m,
        den,
        acc,
        head_mask,
        SCALE * RCP_LN2,
        smem,
        MF,
        MF,
        gl.DotOperandLayout(1, MF, 8),
        gl.DotOperandLayout(1, MF, 8),
        gl.DotOperandLayout(0, MF, 8),
        512,
        0,
        512,
        16,
        BLOCK,
        H % 16 == 0,
        True,
        True,
    )


@gluon.jit
def _compact_tile(
    Cache,
    Indices,
    t,
    start,
    length,
    STRIDE: gl.constexpr,
    CAPACITY: gl.constexpr,
    PAGE: gl.constexpr,
    CACHE_STRIDE: gl.constexpr,
    FP4: gl.constexpr,
    BUFFER: gl.constexpr,
    G: gl.constexpr,
    BLOCK: gl.constexpr,
):
    col = start + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, G))
    slot = gl.load(Indices + t * STRIDE + col, col < length, -1)
    valid = (col < length) & (slot >= 0) & (slot < CAPACITY)
    kv = _read_compact_gluon(
        Cache, gl.where(valid, slot, 0), valid, PAGE, CACHE_STRIDE, FP4, BUFFER, G
    )
    return kv, valid


@gluon.jit
def _compact_attention_kernel(
    Q,
    K,
    E,
    I,
    EI,
    L,
    EL,
    Acc,
    Max,
    Sum,
    Out,
    Sink,
    Freqs,
    Positions,
    HAS_SINK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    FREQ_STRIDE: gl.constexpr,
    QS: gl.constexpr,
    QH: gl.constexpr,
    H: gl.constexpr,
    KS: gl.constexpr,
    ES: gl.constexpr,
    KP: gl.constexpr,
    EP: gl.constexpr,
    KN: gl.constexpr,
    EN: gl.constexpr,
    IS: gl.constexpr,
    EIS: gl.constexpr,
    NK: gl.constexpr,
    NE: gl.constexpr,
    KFP4: gl.constexpr,
    EFP4: gl.constexpr,
    KBUFFER: gl.constexpr,
    EBUFFER: gl.constexpr,
    SPLITS: gl.constexpr,
    SCALE: gl.constexpr,
    BLOCK: gl.constexpr,
):
    t, split, hg = gl.program_id(0).to(gl.int64), gl.program_id(1), gl.program_id(2)
    MF: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    G: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [1, 4], [1, 0])
    QG: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    hq = hg * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, QG))
    dq = gl.arange(0, 512, layout=gl.SliceLayout(0, QG))
    q = gl.load(Q + t * QS + hq[:, None] * QH + dq[None, :], hq[:, None] < H, 0)
    qdot = gl.convert_layout(q, gl.DotOperandLayout(0, MF, 8))
    h = hg * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, MF))
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, MF))
    m = gl.full((16,), -float("inf"), gl.float32, layout=gl.SliceLayout(1, MF))
    den = gl.zeros((16,), gl.float32, layout=gl.SliceLayout(1, MF))
    acc = gl.zeros((16, 512), gl.float32, layout=MF)
    smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [BLOCK, 512],
        gl.PaddedSharedLayout.with_identity_for([[512, 8]], [BLOCK, 512], [1, 0]),
    )
    for segment in gl.static_range(2):
        if segment == 0 or NE > 0:
            if segment == 0:
                indices, lengths, width, stride, capacity = I, L, NK, IS, KN
                cache, page, cache_stride, fp4, buffer = K, KP, KS, KFP4, KBUFFER
            else:
                indices, lengths, width, stride, capacity = EI, EL, NE, EIS, EN
                cache, page, cache_stride, fp4, buffer = E, EP, ES, EFP4, EBUFFER
            length = gl.minimum(gl.load(lengths + t), width)
            if split * BLOCK < length:
                kv, valid = _compact_tile(
                    cache,
                    indices,
                    t,
                    split * BLOCK,
                    length,
                    stride,
                    capacity,
                    page,
                    cache_stride,
                    fp4,
                    buffer,
                    G,
                    BLOCK,
                )
                for start in range((split + SPLITS) * BLOCK, length, SPLITS * BLOCK):
                    next_kv, next_valid = _compact_tile(
                        cache,
                        indices,
                        t,
                        start,
                        length,
                        stride,
                        capacity,
                        page,
                        cache_stride,
                        fp4,
                        buffer,
                        G,
                        BLOCK,
                    )
                    m, den, acc = _compact_qkpv(
                        kv, valid, qdot, m, den, acc, h < H, smem, MF, H, BLOCK, SCALE
                    )
                    kv, valid = next_kv, next_valid
                m, den, acc = _compact_qkpv(
                    kv, valid, qdot, m, den, acc, h < H, smem, MF, H, BLOCK, SCALE
                )
    if SPLITS == 1:
        if HAS_SINK:
            sink = gl.load(Sink + h, h < H, -float("inf"))
            scaled_max = m * (SCALE * RCP_LN2)
            final_max = gl.maximum(scaled_max, sink * RCP_LN2)
            weight = gl.exp2(scaled_max - final_max)
            den = den * weight + gl.exp2(sink * RCP_LN2 - final_max)
            acc = acc * weight[:, None]
        out = (acc / den[:, None]).to(gl.bfloat16)
        out = gl.convert_layout(out, QG)
        if ROPE_DIM > 0:
            pos = gl.load(Positions + t)
            is_rope = dq >= 512 - ROPE_DIM
            freq_index = ((dq - (512 - ROPE_DIM)) // 2) * 2
            cos = gl.load(Freqs + pos * FREQ_STRIDE + freq_index, is_rope, 0)
            sin = gl.load(Freqs + pos * FREQ_STRIDE + freq_index + 1, is_rope, 0)
            x = out.to(gl.float32)
            even, odd = gl.split(gl.reshape(x, (16, 256, 2)))
            swapped = gl.reshape(gl.join(odd, -even), (16, 512))
            rotated = gl.fma(x, cos[None, :], swapped * sin[None, :])
            out = gl.where(is_rope[None, :], rotated.to(gl.bfloat16), out)
        gl.store(Out + (t * H + hq[:, None]) * 512 + dq[None, :], out, hq[:, None] < H)
    else:
        off = (t * SPLITS + split) * H + h
        gl.store(Max + off, m * (SCALE * RCP_LN2), h < H)
        gl.store(Sum + off, den, h < H)
        gl.store(Acc + off[:, None] * 512 + d[None, :], acc, h[:, None] < H)


def compact_attention_hip(
    q,
    cache,
    indices,
    lengths,
    sink,
    *,
    extra_cache=None,
    extra_indices=None,
    extra_lengths=None,
    softmax_scale=512**-0.5,
    splits=None,
    inv_rope=None,
):
    """Packed caches have shape [pages, page_size, 1, bytes_per_token]."""
    assert q.ndim == 3 and q.shape[-1] == 512 and q.dtype == torch.bfloat16
    assert q.stride(-1) == 1
    assert cache.ndim == 4 and cache.shape[2] == 1 and cache.shape[-1] in (528, 288)
    n, h, _ = q.shape
    if n == 0:
        return torch.empty_like(q)
    indices = indices.reshape(n, -1)
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    if extra_cache is None:
        assert extra_indices is None and extra_lengths is None
        extra_cache, extra_indices, extra_lengths = cache, indices, lengths
        ne = 0
    else:
        assert extra_cache.shape[-1] in (528, 288)
        extra_indices = extra_indices.reshape(n, -1)
        assert extra_indices.stride(-1) == 1 and extra_lengths.is_contiguous()
        ne = extra_indices.shape[1]
    if splits is None:
        splits = min(
            8,
            triton.cdiv(indices.shape[1] + ne, 64),
            triton.cdiv(256, max(1, n * triton.cdiv(h, 16))),
        )
    assert splits > 0
    out = torch.empty((n, h, 512), dtype=q.dtype, device=q.device) if splits == 1 else q
    if inv_rope is not None:
        freqs, positions = inv_rope
        assert freqs.dtype == torch.float32 and freqs.stride(1) == 1
        assert positions.shape == (n,)
        rope_dim = freqs.shape[1]
        assert 0 < rope_dim <= 512 and rope_dim % 2 == 0
    else:
        freqs = positions = out
        rope_dim = 0
    if splits == 1:
        acc = maximum = denominator = out
    else:
        acc = torch.empty((n, splits, h, 512), dtype=torch.float32, device=q.device)
        maximum = torch.empty((n, splits, h), dtype=torch.float32, device=q.device)
        denominator = torch.empty_like(maximum)
    _compact_attention_kernel[(n, splits, triton.cdiv(h, 16))](
        q,
        cache.view(torch.uint8),
        extra_cache.view(torch.uint8),
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        acc,
        maximum,
        denominator,
        out,
        sink if sink is not None else out,
        freqs,
        positions,
        HAS_SINK=sink is not None,
        ROPE_DIM=rope_dim,
        FREQ_STRIDE=freqs.stride(0),
        QS=q.stride(0),
        QH=q.stride(1),
        H=h,
        KS=cache.stride(0),
        ES=extra_cache.stride(0),
        KP=cache.shape[1],
        EP=extra_cache.shape[1],
        KN=cache.shape[0] * cache.shape[1],
        EN=extra_cache.shape[0] * extra_cache.shape[1],
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        NK=indices.shape[1],
        NE=ne,
        KFP4=cache.shape[-1] == 288,
        EFP4=extra_cache.shape[-1] == 288,
        KBUFFER=cache.shape[0] * cache.stride(0) < 2**31,
        EBUFFER=extra_cache.shape[0] * extra_cache.stride(0) < 2**31,
        SPLITS=splits,
        SCALE=softmax_scale,
        BLOCK=64,
        num_warps=4,
    )
    if splits == 1:
        return out
    return aiter_sparse_split_reduce(
        acc, maximum, denominator, sink, q.dtype, inv_rope=inv_rope
    )
