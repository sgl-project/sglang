"""SM100 small-batch paged attention with the heads on the MMA N dimension; the
caller applies the inverse RoPE to the result."""

from typing import Optional

import torch
import triton
import triton.language as tl

from .kv_layout import KVLayout

LAYOUT = KVLayout.V4
MAX_BATCH = 8
NUM_HEADS = 16
HEAD_DIM = 512
SOFTMAX_SCALE = HEAD_DIM**-0.5


def can_use_swapab_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    extra_kv: Optional[torch.Tensor],
    num_heads: int,
    head_dim_v: int,
    softmax_scale: float,
) -> bool:
    """The caller adds the SM100 and single-query forward-mode gates."""
    return (
        0 < q.shape[0] <= MAX_BATCH
        and num_heads == NUM_HEADS
        and q.dtype == torch.bfloat16
        and q.shape[-1] == HEAD_DIM
        and head_dim_v == HEAD_DIM
        and softmax_scale == SOFTMAX_SCALE
        and kv.shape[-1] == LAYOUT.bytes_per_token
        and (extra_kv is None or extra_kv.shape[-1] == LAYOUT.bytes_per_token)
    )


@triton.jit
def _combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
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
):
    """V4-layout attention on 16 heads; `extra_*` is a second slot range appended
    to each request's keys, and the attention sink is folded in exactly once."""
    from .decode_attention_sm100_gluon import partial_gluon

    block = 64
    b, h, d = q.shape[0], q.shape[-2], q.shape[-1]
    assert q.ndim in (3, 4) and (q.ndim == 3 or q.shape[1] == 1)
    assert 0 < b <= MAX_BATCH and h == NUM_HEADS and d == HEAD_DIM
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    assert kv.shape[-1] == LAYOUT.bytes_per_token
    assert kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    assert indices.stride(-1) == 1 and lengths.is_contiguous()
    assert sink.stride(0) == 1 and sink.numel() >= h
    nk = indices.shape[-1]
    ne = 0 if extra_indices is None else extra_indices.shape[-1]
    assert block in (32, 64, 128) and nk > 0
    if extra_kv is None:
        assert extra_indices is None and extra_lengths is None
        extra_kv, extra_indices, extra_lengths = kv, indices, lengths
    else:
        assert extra_kv.shape[-1] == LAYOUT.bytes_per_token
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
        DATA_BYTES=LAYOUT.data_bytes,
        SCALE_BYTES=LAYOUT.scale_bytes,
        TILE=LAYOUT.tile_size,
        num_warps=4,
    )
    bd = 64 if ne else 512
    _combine[(b, h, triton.cdiv(512, bd))](
        partial,
        maximum,
        sums,
        sink,
        out,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=h,
        BD=bd,
        num_warps=4,
    )
    return out
