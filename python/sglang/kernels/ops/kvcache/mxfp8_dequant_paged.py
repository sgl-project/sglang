"""Dequantize the MXFP8 KV cache (e4m3 data, interleaved UE8M0 scales) to BF16.

The pool stores a 128-token page's scales for one head as 128 32-bit words; slot
t's word is (t % 32) * 4 + t // 32 and its byte j scales elements 32j..32j+31
(store_sf_interleaved). A UE8M0 byte e is 2 ** (e - 127), with e = 255 NaN.
"""

import torch
import triton
import triton.language as tl

MXFP8_PAGE = 128
_HEAD_DIM = 128


def mxfp8_scale_words(scale: torch.Tensor, head_dim: int) -> torch.Tensor:
    """The pool's scales (pages, heads, 32, 4, 4) as int32 words (pages, heads, 128)."""
    assert head_dim == _HEAD_DIM and scale.is_contiguous()
    assert scale.shape[2:] == (32, MXFP8_PAGE // 32, head_dim // 32)
    return scale.view(scale.shape[0], scale.shape[1], -1).view(torch.int32)


@triton.jit
def _ue8m0_to_f32(e):
    bits = tl.where(e == 0, 0x400000, tl.where(e == 255, 0x7FC00000, e << 23))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def mxfp8_dequant_rows(x, Words, slot, mask, D: tl.constexpr):
    """x (rows, D) e4m3 at in-page slots `slot`; Words points at the (page, head)
    scale words. Returns the rows in BF16; masked rows read scale 2 ** -127."""
    word = tl.load(Words + (slot % 32) * 4 + slot // 32, mask, other=0)
    shift = tl.arange(0, D) // 32 * 8
    e = (word[:, None] >> shift[None, :]) & 0xFF
    y = (x.to(tl.float32) * _ue8m0_to_f32(e)).to(tl.bfloat16)
    # Triton sizes a dot operand's k-width from the narrowest load feeding it;
    # the impure copy hides the 8-bit load, so a dot sums as on a BF16 cache.
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [y], dtype=tl.bfloat16, is_pure=False, pack=2
    )


@triton.jit
def _dequant_pages(
    K,
    V,
    KWords,
    VWords,
    DK,
    DV,
    PageTable,
    SeqLens,
    CuSeqLensQ,
    SP,
    SK0: tl.constexpr,
    SV0: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SDK0: tl.constexpr,
    SDV0: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0) // HEADS
    h = tl.program_id(0) % HEADS
    seqlen = tl.load(SeqLens + b)
    first = 0
    if WINDOW >= 0:
        qlen = tl.load(CuSeqLensQ + b + 1) - tl.load(CuSeqLensQ + b)
        first = tl.maximum(seqlen - qlen - WINDOW, 0)
    d = tl.arange(0, D)
    for j in range(
        first // PAGE + tl.program_id(1), tl.cdiv(seqlen, PAGE), tl.num_programs(1)
    ):
        page = tl.load(PageTable + b.to(tl.int64) * SP + j).to(tl.int64)
        k_words = KWords + page * SW0 + h * SW1
        v_words = VWords + page * SW0 + h * SW1
        for sub in tl.static_range(PAGE // BLOCK):
            slot = sub * BLOCK + tl.arange(0, BLOCK)
            row = (page * PAGE + slot)[:, None]
            k = tl.load(K + row * SK0 + h * D + d[None, :])
            k = mxfp8_dequant_rows(k, k_words, slot, slot < PAGE, D)
            tl.store(DK + row * SDK0 + h * D + d[None, :], k)
            v = tl.load(V + row * SV0 + h * D + d[None, :])
            v = mxfp8_dequant_rows(v, v_words, slot, slot < PAGE, D)
            tl.store(DV + row * SDV0 + h * D + d[None, :], v)


def mxfp8_dequant_paged(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    page_table: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window_left: int = -1,
) -> None:
    """Dequantize every page a paged attention call reads into k_out/v_out at the
    same slots; whole pages, past a sequence's end included. k/v_cache e4m3 and
    k/v_out BF16, (slots, heads, 128); k/v_scale the pool's interleaved scales;
    page_table (batch, pages), 128-token pages; seqlens (batch,) keys;
    cu_seqlens_q (batch + 1,). window_left >= 0 skips pages wholly before every
    query's window. The grid depends only on shapes, so CUDA graphs capture it."""
    slots, heads, head_dim = k_cache.shape
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k_out.dtype == v_out.dtype == torch.bfloat16
    assert k_out.shape[0] >= slots and k_out.shape[1:] == (heads, head_dim)
    assert v_cache.shape == k_cache.shape and v_out.shape == k_out.shape
    for t in (k_cache, v_cache, k_out, v_out):
        assert t.stride(1) == head_dim and t.stride(2) == 1
    k_words = mxfp8_scale_words(k_scale, head_dim)
    v_words = mxfp8_scale_words(v_scale, head_dim)
    assert k_words.stride() == v_words.stride()
    batch = seqlens.shape[0]
    if batch == 0:
        return
    sms = torch.cuda.get_device_properties(k_cache.device).multi_processor_count
    pages = max(1, min(page_table.shape[1], triton.cdiv(4 * sms, batch * heads)))
    _dequant_pages[(batch * heads, pages)](
        k_cache,
        v_cache,
        k_words,
        v_words,
        k_out,
        v_out,
        page_table,
        seqlens,
        cu_seqlens_q,
        page_table.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        k_words.stride(0),
        k_words.stride(1),
        k_out.stride(0),
        v_out.stride(0),
        heads,
        head_dim,
        MXFP8_PAGE,
        window_left,
        64,
        num_warps=4,
    )
