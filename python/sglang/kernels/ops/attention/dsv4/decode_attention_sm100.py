"""SM100 small-batch paged attention with the heads on the MMA N dimension; the
inverse RoPE is applied by the caller, or by the combine when inv_rope is given."""

from typing import Optional

import torch
import triton
import triton.language as tl

from .kv_layout import KVLayout, is_valid_kv_layout_pair

# Every paged layout the V4 family reaches this kernel with, main cache and
# compressed cache alike.
KV_LAYOUTS = (KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4)
MAX_BATCH = 8
NUM_HEADS = 16
HEAD_DIM = 512
SOFTMAX_SCALE = HEAD_DIM**-0.5


def _layout_of_bytes(bytes_per_token: int) -> Optional[KVLayout]:
    """The paged layout whose token row is this many bytes, or None.

    A 528-byte row resolves to V4.1 here rather than to V3.2-no-RoPE: only the V4
    family reaches this kernel, and FlashMLA disambiguates the same collision with
    the kv_format argument the caller passes it.
    """
    for layout in KV_LAYOUTS:
        if layout.bytes_per_token == bytes_per_token:
            return layout
    return None


def can_use_swapab_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    extra_kv: Optional[torch.Tensor],
    num_heads: int,
    head_dim_v: int,
    softmax_scale: float,
) -> bool:
    """The caller adds the SM100 and single-query forward-mode gates."""
    main = _layout_of_bytes(kv.shape[-1])
    if main is None:
        return False
    extra = main if extra_kv is None else _layout_of_bytes(extra_kv.shape[-1])
    return (
        0 < q.shape[0] <= MAX_BATCH
        and num_heads == NUM_HEADS
        and q.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM
        and head_dim_v == HEAD_DIM
        and softmax_scale == SOFTMAX_SCALE
        and extra is not None
        and is_valid_kv_layout_pair(main, extra)
    )


@triton.jit
def inverse_rope_tail(
    out, d, FREQS, pos, freq_stride, HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr
):
    """out (head dims d) with its last ROPE_DIM dims inverse-rotated at position pos. It
    rotates the rounded values, as the model's standalone RoPE kernel reads them."""
    x = out.to(tl.float32)
    is_rope = d >= HEAD_DIM - ROPE_DIM
    freq_index = ((d - (HEAD_DIM - ROPE_DIM)) // 2) * 2
    cos = tl.load(FREQS + pos * freq_stride + freq_index, is_rope, 0)
    sin = tl.load(FREQS + pos * freq_stride + freq_index + 1, is_rope, 0)
    signed = tl.where(d % 2 == 0, -x * sin, x * sin)
    swapped = tl.reshape(tl.flip(tl.reshape(signed, (d.shape[0] // 2, 2)), 1), d.shape)
    rotated = tl.fma(x, cos, swapped)
    return tl.where(is_rope, rotated.to(out.dtype), out)


@triton.jit
def _combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    FREQS,
    POSITIONS,
    FREQ_STRIDE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    BD: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, ST), tile * BD + tl.arange(0, BD)
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    sink = tl.load(SINK + h)
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * 512 + d[None, :],
        t[:, None] < NT,
        0,
    )
    out = tl.sum(vals * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    out = out.to(OUT.dtype.element_ty)
    if ROPE_DIM > 0:
        pos = tl.load(POSITIONS + b)
        out = inverse_rope_tail(out, d, FREQS, pos, FREQ_STRIDE, 512, ROPE_DIM)
    tl.store(OUT + (b * H + h) * 512 + d, out)


def swapab_attention(
    q,
    kv,
    indices,
    lengths,
    sink,
    extra_kv=None,
    extra_indices=None,
    extra_lengths=None,
    inv_rope=None,
    partial_gluon=None,
):
    """Paged V4-family attention on 16 heads; `extra_*` is a second slot range
    appended to each request's keys, and the attention sink is folded in exactly
    once. `partial_gluon` replaces the SM100 split-KV kernel (swapab_gluon_hip
    passes gfx950's)."""
    if partial_gluon is None:
        from .decode_attention_sm100_gluon import partial_gluon

    block = 64
    b, h, d = q.shape[0], q.shape[-2], q.shape[-1]
    assert q.ndim in (3, 4) and (q.ndim == 3 or q.shape[1] == 1)
    assert 0 < b <= MAX_BATCH and h == NUM_HEADS and d == HEAD_DIM
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    main = _layout_of_bytes(kv.shape[-1])
    assert main is not None, (
        f"unsupported main-cache row of {kv.shape[-1]} bytes; expected one of "
        f"{[layout.bytes_per_token for layout in KV_LAYOUTS]}"
    )
    assert kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    assert sink.stride(0) == 1 and sink.numel() >= h
    nk = indices.shape[-1]
    ne = 0 if extra_indices is None else extra_indices.shape[-1]
    assert block in (32, 64, 128) and nk > 0
    if extra_kv is None:
        assert extra_indices is None and extra_lengths is None
        extra_kv, extra_indices, extra_lengths = kv, indices, lengths
        extra = main
    else:
        # The compressed cache carries its own layout: under V4.1 a layer's
        # ratio-1/2 latents are fp4 while ratios 4/128 stay fp8, so the two
        # caches differ within one forward.
        extra = _layout_of_bytes(extra_kv.shape[-1])
        assert extra is not None and is_valid_kv_layout_pair(main, extra), (
            f"invalid cache pair: main {main.value} of {kv.shape[-1]} bytes and "
            f"extra of {extra_kv.shape[-1]} bytes"
        )
        assert extra_kv.dtype in (torch.uint8, torch.float8_e4m3fn)
        assert extra_indices.stride(-1) == 1 and extra_lengths.is_contiguous()
    kv, extra_kv = kv.view(torch.uint8), extra_kv.view(torch.uint8)
    kt = triton.cdiv(nk, block)
    nt = kt + triton.cdiv(ne, block)
    partial = torch.empty((b, nt, h, 512), dtype=torch.float32, device=q.device)
    maximum = torch.empty((b, nt, h), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    out = torch.empty((b, h, 512), dtype=q.dtype, device=q.device)
    partial_gluon[(b, nt)](
        q,
        kv,
        extra_kv,
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        partial,
        maximum,
        sums,
        QS=q.stride(0),
        QH=q.stride(-2),
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        KP=kv.shape[1],
        KS=kv.stride(0),
        EP=extra_kv.shape[1],
        ES=extra_kv.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=block,
        H=h,
        SCALE=SOFTMAX_SCALE,
        KTOKENS=kv.shape[0] * kv.shape[1],
        ETOKENS=extra_kv.shape[0] * extra_kv.shape[1],
        COMPENSATE=True,
        SWAP_AB=True,
        DATA_BYTES=main.data_bytes,
        SCALE_BYTES=main.scale_bytes,
        TILE=main.tile_size,
        E_DATA_BYTES=extra.data_bytes,
        E_SCALE_BYTES=extra.scale_bytes,
        E_TILE=extra.tile_size,
        IS_V41=main is not KVLayout.V4,
        IS_FP4=main.is_fp4,
        E_IS_V41=extra is not KVLayout.V4,
        E_IS_FP4=extra.is_fp4,
        num_warps=4,
    )
    if inv_rope is not None:
        freqs, positions = inv_rope
        assert freqs.dtype == torch.float32 and freqs.stride(1) == 1
        assert positions.shape == (b,)
        rope_dim = freqs.shape[1]
        assert 0 < rope_dim <= 512 and rope_dim % 2 == 0
    else:
        freqs = positions = out
        rope_dim = 0
    bd = 64 if ne else 512
    _combine[(b, h, triton.cdiv(512, bd))](
        partial,
        maximum,
        sums,
        sink,
        out,
        freqs,
        positions,
        FREQ_STRIDE=freqs.stride(0),
        ROPE_DIM=rope_dim,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=h,
        BD=bd,
        num_warps=4,
    )
    return out
