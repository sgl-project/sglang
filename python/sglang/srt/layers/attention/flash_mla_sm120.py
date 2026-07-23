"""SM120 FlashMLA sparse decode implementation.

On SM120 (Blackwell Desktop / RTX PRO 6000) the flash_mla CUDA kernel
is not available, so this module provides alternative implementations:

- A fused Triton kernel (default, ``SGLANG_SM120_TRITON_FLASHMLA=1``)
- A pure-PyTorch fallback (``SGLANG_SM120_TRITON_FLASHMLA=0``)

The FP8 KV cache uses a page-internal layout where NOPE+ROPE data has
stride (nope_dim + rope_dim*2) per token, and scales are stored in a
separate region at the end of each page.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Page layout constants for DSv4-Flash (MODEL1):
#   nope_dim = 448, rope_dim = 64, quantize_block_size = 64
#   nope_rope_stride = 448 + 64*2 = 576 bytes per token
#   scale_stride = ceil(448/64) + 1 = 8 bytes per token (7 scales + 1 pad)
#   bytes_per_token = 448 + 128 + 8 = 584
#   page_bytes = ceil_div(page_size * 584, 576) * 576

_NOPE_DIM = 448
_ROPE_DIM = 64
_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576
_TILE_SIZE = 64
_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
_SCALE_STRIDE = _NUM_TILES + 1  # 8 (7 scales + 1 pad)
_D = _NOPE_DIM + _ROPE_DIM  # 512


def _gather_and_dequant(k_cache, indices, page_size):
    """Gather KV entries from the paged buffer using correct page-internal addressing.

    Args:
        k_cache: (num_pages, page_size, 1, bytes_per_token) float8_e4m3fn
                 Non-contiguous view of the raw page buffer.
        indices: (...) int32/int64, token-level indices. -1 = invalid.
        page_size: tokens per page (256)

    Returns:
        kv: (..., _D) bfloat16, dequantized KV vectors
    """
    idx_shape = indices.shape
    flat_idx = indices.reshape(-1)  # (N,)
    N = flat_idx.shape[0]
    device = k_cache.device

    # Page-level addressing
    page_bytes = k_cache.stride(0)  # actual byte stride between pages
    pages = flat_idx // page_size
    offsets = flat_idx % page_size

    # Clamp invalid indices
    safe_pages = pages.clamp(min=0)
    safe_offsets = offsets.clamp(min=0)

    # Access raw buffer as uint8 — use as_strided to get full page view
    num_pages = k_cache.shape[0]
    raw_pages = k_cache.as_strided(
        (num_pages, page_bytes),
        (page_bytes, 1),
    ).view(
        torch.uint8
    )  # (num_pages, page_bytes) uint8
    # Note: float8_e4m3fn and uint8 are both 1 byte, view is safe

    # Compute byte offsets within each page
    # NOPE: page[safe_page, safe_offset * 576 + 0:448]
    # ROPE: page[safe_page, safe_offset * 576 + 448:576]
    # SCALES: page[safe_page, page_size * 576 + safe_offset * 8 + 0:7]

    nope_base = safe_offsets * _NOPE_ROPE_STRIDE  # (N,)
    nope_offsets = nope_base.unsqueeze(-1) + torch.arange(
        _NOPE_DIM, device=device, dtype=torch.long
    )  # (N, 448)

    rope_base = nope_base + _NOPE_DIM  # (N,)
    rope_offsets = rope_base.unsqueeze(-1) + torch.arange(
        _ROPE_DIM * 2, device=device, dtype=torch.long
    )  # (N, 128)

    scale_section_offset = page_size * _NOPE_ROPE_STRIDE  # 147456
    scale_base = scale_section_offset + safe_offsets * _SCALE_STRIDE  # (N,)
    scale_offsets = scale_base.unsqueeze(-1) + torch.arange(
        _NUM_TILES, device=device, dtype=torch.long
    )  # (N, 7)

    # Gather bytes per page — use advanced indexing
    # raw_pages[safe_pages, nope_offsets] → (N, 448)
    page_idx_nope = safe_pages.unsqueeze(-1).expand_as(nope_offsets)
    nope_bytes = raw_pages[page_idx_nope, nope_offsets]  # (N, 448) uint8

    page_idx_rope = safe_pages.unsqueeze(-1).expand_as(rope_offsets)
    rope_bytes = raw_pages[page_idx_rope, rope_offsets]  # (N, 128) uint8

    page_idx_scale = safe_pages.unsqueeze(-1).expand_as(scale_offsets)
    scale_bytes = raw_pages[page_idx_scale, scale_offsets]  # (N, 7) uint8

    # Reinterpret dtypes
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (N, 448)
    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (N, 64)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (N, 7)

    # Dequantize: nope_tile * scale_tile → bf16 (vectorized)
    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)
    result[:, :_NOPE_DIM] = (
        (
            nope_fp8.view(N, _NUM_TILES, _TILE_SIZE).float()
            * scale_e8m0.view(N, _NUM_TILES, 1).float()
        )
        .view(N, _NOPE_DIM)
        .to(torch.bfloat16)
    )
    result[:, _NOPE_DIM:] = rope_bf16

    return result.reshape(*idx_shape, _D)


def _sm120_sparse_decode_fwd(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache=None,
    extra_indices=None,
    extra_topk_length=None,
):
    B, s_q, H_q, D_qk = q.shape
    num_pages, page_size, H_k, bpt = k_cache.shape
    topk = indices.shape[-1]

    invalid_mask = indices < 0
    safe_indices = indices.clamp(min=0)

    if topk_length is not None:
        topk_range = torch.arange(topk, device=topk_length.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (topk_range >= topk_length.view(B, 1, 1))

    # Gather and dequantize using page-aware addressing
    gathered_kv = _gather_and_dequant(k_cache, safe_indices, page_size)

    if extra_k_cache is not None and extra_indices is not None:
        extra_topk = extra_indices.shape[-1]
        extra_page_size = extra_k_cache.shape[1]
        extra_invalid = extra_indices < 0
        extra_safe = extra_indices.clamp(min=0)
        if extra_topk_length is not None:
            extra_range = torch.arange(
                extra_topk, device=extra_topk_length.device
            ).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (
                extra_range >= extra_topk_length.view(B, 1, 1)
            )
        extra_kv = _gather_and_dequant(extra_k_cache, extra_safe, extra_page_size)
        gathered_kv = torch.cat([gathered_kv, extra_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)

    gathered_kv[invalid_mask] = 0.0

    q_f = q.float()
    kv_f = gathered_kv.float()
    kv_d = kv_f.shape[-1]
    if D_qk != kv_d:
        q_f = q_f[..., :kv_d]

    scores = torch.einsum("bshd,bstd->bsht", q_f, kv_f) * softmax_scale
    scores.masked_fill_(invalid_mask.unsqueeze(2).expand_as(scores), float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)

    if attn_sink is not None:
        lse_for_out = torch.logsumexp(
            torch.stack([lse, attn_sink.view(1, 1, H_q).expand_as(lse)], dim=0), dim=0
        )
    else:
        lse_for_out = lse.clone()

    lonely = lse == float("-inf")
    lse_for_out[lonely] = float("inf")
    weights = torch.exp(scores - lse_for_out.unsqueeze(-1))
    out = torch.einsum("bsht,bstv->bshv", weights, kv_f[..., :head_dim_v])
    out[lonely.unsqueeze(-1).expand_as(out)] = 0.0

    return out.to(torch.bfloat16), lse.permute(0, 2, 1)


# SM120 FlashMLA: default FlashInfer (CUTLASS SM120 sparse MLA decode).
# Override with SGLANG_SM120_FLASHMLA_BACKEND=triton|torch to force fallback.
_sm120_default_backend = envs.SGLANG_SM120_FLASHMLA_BACKEND.get()


def flash_mla_with_kvcache_sm120(**kwargs):
    """SM120 FlashMLA sparse decode entry point.

    Dispatches to FlashInfer (default if available), Triton, or PyTorch fallback.
    """
    q = kwargs["q"]
    k_cache = kwargs["k_cache"]
    indices = kwargs["indices"]
    topk_length = kwargs.get("topk_length")
    attn_sink = kwargs.get("attn_sink")
    head_dim_v = kwargs["head_dim_v"]
    softmax_scale = kwargs.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    extra_k_cache = kwargs.get("extra_k_cache")
    extra_indices = kwargs.get("extra_indices_in_kvcache")
    extra_topk_length = kwargs.get("extra_topk_length")

    if _sm120_default_backend == "flashinfer":
        return _flash_mla_flashinfer(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )

    if _sm120_default_backend == "triton":
        from sglang.srt.layers.attention.flash_mla_sm120_triton import (
            flash_mla_sparse_decode_triton,
        )

        out, lse = flash_mla_sparse_decode_triton(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            head_dim_v,
            softmax_scale,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
        )
        return (out, lse)

    out, lse = _sm120_sparse_decode_fwd(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        head_dim_v,
        softmax_scale,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
    )
    return (out, lse)


# --- Page-split utilities: pbs=256 → pbs=64 ---
# SGLang SWA KV cache footer layout per 256-token page:
#   [data: 256 * 576 bytes] [scale: 256 * 8 bytes] [padding]
# FlashInfer decode_dsv4 expects per 64-token page:
#   [data: 64 * 576 bytes] [scale: 64 * 8 bytes] [padding to 37440]
_PBS_SRC = 256  # SGLang physical page size
_PBS_DST = 64  # FlashInfer page_block_size
_NOPE_ROPE_STRIDE = 576  # bytes per token for nope+rope
_SCALE_STRIDE = 8  # bytes per token for scale (7 + 1 pad)
_BYTES_PER_DST_PAGE = (
    _PBS_DST * _NOPE_ROPE_STRIDE + _PBS_DST * _SCALE_STRIDE
)  # 64*576 + 64*8 = 37376 + 512 = 37888
# Padded to 576 alignment

_BYTES_PER_DST_PAGE_PADDED = math.ceil(_BYTES_PER_DST_PAGE / 576) * 576  # 37440


@triton.jit
def _page_split_kernel(
    src_ptr,
    dst_ptr,
    N_pages,
    src_stride0: tl.constexpr,
    dst_stride0: tl.constexpr,
    DATA_PER_SUB: tl.constexpr,  # 64 * 576 = 36864
    SCALE_PER_SUB: tl.constexpr,  # 64 * 8 = 512
    SRC_SCALE_OFF: tl.constexpr,  # 256 * 576 = 147456
    DST_SCALE_OFF: tl.constexpr,  # 64 * 576 = 36864
    RATIO: tl.constexpr,  # 4
    BLOCK_SIZE: tl.constexpr,
):
    """Fused page-split: copy data+scale for all sub-pages in one kernel."""
    pid = tl.program_id(0)
    page_idx = pid // RATIO
    sub = pid % RATIO

    if page_idx >= N_pages:
        return

    src_base = src_ptr + page_idx * src_stride0
    dst_base = dst_ptr + (page_idx * RATIO + sub) * dst_stride0

    # Copy data region: DATA_PER_SUB bytes from src offset sub*DATA_PER_SUB
    data_src_off = sub * DATA_PER_SUB
    for start in tl.range(0, DATA_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DATA_PER_SUB
        vals = tl.load(src_base + data_src_off + offs, mask=mask)
        tl.store(dst_base + offs, vals, mask=mask)

    # Copy scale region: SCALE_PER_SUB bytes
    scale_src_off = SRC_SCALE_OFF + sub * SCALE_PER_SUB
    for start in tl.range(0, SCALE_PER_SUB, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < SCALE_PER_SUB
        vals = tl.load(src_base + scale_src_off + offs, mask=mask)
        tl.store(dst_base + DST_SCALE_OFF + offs, vals, mask=mask)


def _split_kv_pages_to_64(kv_u8: torch.Tensor, src_pbs: int) -> torch.Tensor:
    """Split pbs=N footer-format pages into pbs=64 footer-format pages.

    Uses a fused Triton kernel to do all sub-page copies in a single launch
    instead of 8 separate copy kernels (4 sub-pages × 2 regions).
    """
    assert src_pbs % _PBS_DST == 0 and src_pbs >= _PBS_DST
    if src_pbs == _PBS_DST:
        return kv_u8

    N = kv_u8.shape[0]
    ratio = src_pbs // _PBS_DST
    num_dst_pages = N * ratio

    from sglang.srt.runtime_context import get_resources

    # Pre-allocated grow-only buffer for page-split output per device.
    dev = kv_u8.device
    buffers = get_resources().buffers
    key = f"flash_mla_sm120_split:{dev}"
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < num_dst_pages:
        buf = torch.empty(
            num_dst_pages,
            _BYTES_PER_DST_PAGE_PADDED,
            dtype=torch.uint8,
            device=dev,
        )
        buffers[key] = buf
    out = buf[:num_dst_pages]

    # Get raw 2D view of source
    src_2d = kv_u8
    if src_2d.ndim == 4:
        src_stride0 = src_2d.stride(0)
        src_2d = torch.as_strided(src_2d, (N, src_stride0), (src_stride0, 1))
    else:
        src_stride0 = src_2d.stride(0)

    grid = (N * ratio,)
    _page_split_kernel[grid](
        src_2d,
        out,
        N,
        src_stride0,
        _BYTES_PER_DST_PAGE_PADDED,
        _PBS_DST * _NOPE_ROPE_STRIDE,  # DATA_PER_SUB = 36864
        _PBS_DST * _SCALE_STRIDE,  # SCALE_PER_SUB = 512
        src_pbs * _NOPE_ROPE_STRIDE,  # SRC_SCALE_OFF = 147456
        _PBS_DST * _NOPE_ROPE_STRIDE,  # DST_SCALE_OFF = 36864
        ratio,  # RATIO = 4
        1024,  # BLOCK_SIZE
    )

    bpt = _NOPE_ROPE_STRIDE + _SCALE_STRIDE  # 584
    return out.as_strided(
        (num_dst_pages, _PBS_DST, 1, bpt),
        (_BYTES_PER_DST_PAGE_PADDED, bpt, bpt, 1),
    )


def _flash_mla_flashinfer(
    q,
    k_cache,
    indices,
    topk_length,
    attn_sink,
    head_dim_v,
    softmax_scale,
    extra_k_cache,
    extra_indices,
    extra_topk_length,
):
    """FlashInfer SM120 sparse MLA via sparse_mla_sm120_decode_dsv4.

    SGLang SWA pool uses page_size=256 (footer format: 256*576 bytes data + 256*8 bytes scale).
    FlashInfer decode_dsv4 fast path requires page_block_size=64 (footer: 64*576 + 64*8).
    We split 256-token pages into 4 virtual 64-token pages.
    Token indices are invariant under page-split (identity mapping).
    """
    from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv4

    B, _, H, D = q.shape  # (batch, 1, num_heads, head_dim)
    dev = q.device

    # --- Page-split: convert pbs=N kv_cache to pbs=64 view ---
    kv_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    src_pbs = k_cache.shape[1] if k_cache.ndim >= 3 else _PBS_SRC
    kv_64 = _split_kv_pages_to_64(kv_u8, src_pbs) if src_pbs != _PBS_DST else kv_u8

    extra_kv_u8 = (
        extra_k_cache.view(torch.uint8)
        if extra_k_cache is not None and extra_k_cache.dtype != torch.uint8
        else extra_k_cache
    )
    extra_kv_64 = extra_kv_u8

    # Indices: no remapping needed (page-split preserves token addressing).
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    extra_idx = (
        extra_indices.squeeze(1)
        if extra_indices is not None and extra_indices.dim() == 3
        else extra_indices
    )

    output = torch.empty(B, H, head_dim_v, dtype=torch.bfloat16, device=dev)
    out_lse = torch.empty(B, H, dtype=torch.float32, device=dev)

    # Pre-allocate split-K scratch for decode-dsv4 fast path.
    topk = idx.shape[-1]
    extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
    _BI = 64
    num_splits = (topk + _BI - 1) // _BI + (
        (extra_topk + _BI - 1) // _BI if extra_topk > 0 else 0
    )
    mid_out = torch.empty(
        B, H, num_splits, head_dim_v, dtype=torch.bfloat16, device=dev
    )
    mid_lse = torch.empty(B, H, num_splits, dtype=torch.float32, device=dev)

    sparse_mla_sm120_decode_dsv4(
        q=q.squeeze(1) if q.ndim == 4 else q,
        kv_cache=kv_64,
        indices=idx,
        mid_out=mid_out,
        mid_lse=mid_lse,
        output=output,
        out_lse=out_lse,
        sm_scale=softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv_64,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
    )

    return (output.unsqueeze(1), None)
