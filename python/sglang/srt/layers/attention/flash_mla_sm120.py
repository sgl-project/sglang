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
import os

import torch

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


# Default SM120 FlashMLA backend: "triton" (optimized) or "torch" (pure-PyTorch fallback).
# Controlled by SGLANG_SM120_TRITON_FLASHMLA env var (1=triton, 0=torch).
_sm120_default_backend = (
    "triton" if os.environ.get("SGLANG_SM120_TRITON_FLASHMLA", "1") == "1" else "torch"
)


def flash_mla_with_kvcache_sm120(**kwargs):
    """SM120 FlashMLA sparse decode entry point.

    Dispatches to the Triton kernel (default) or PyTorch fallback.
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
