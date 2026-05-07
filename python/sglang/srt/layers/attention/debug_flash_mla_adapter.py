"""Flash MLA adapter with SM120 PyTorch fallback for decode attention.

SM120 (RTX PRO 6000 Blackwell, CC 12.0) does not support flash_mla CUDA kernels
(sm_90a/sm_100f cubins are incompatible). This module provides a pure PyTorch
fallback for the sparse decode path.

Optimizations applied:
  - Zero-copy raw_buf view when k_cache is already contiguous
  - Batched gather (single-pass for all NoPE/RoPE/scale bytes)
  - bf16 tensor core matmul for Q@K^T and attn@V

SWA KV cache layout per page (page_size=256 tokens, dtype=float8_e4m3fn/uint8):
  Two-section layout:
    Section A: NoPE FP8 (448 bytes) + RoPE BF16 (128 bytes) per token at t*576
    Section B: UE8M0 scales (7 bytes + 1 pad) per token at page_size*576 + t*8
"""

import importlib.util
import os

import torch

_SM120 = None
_use_triton_gather = None
_use_triton_flashmla = None


def _should_use_triton_gather():
    """Check if Triton fused gather is available for SM120."""
    global _use_triton_gather
    if _use_triton_gather is None:
        _use_triton_gather = (
            importlib.util.find_spec(
                "sglang.srt.layers.attention.fused_kv_gather_triton"
            )
            is not None
        )
    return _use_triton_gather


def _should_use_triton_flashmla():
    """Check if Triton tiled FlashMLA kernel should be used on SM120.

    Controlled by SGLANG_SM120_TRITON_FLASHMLA=1 env var for A/B testing.
    """
    global _use_triton_flashmla
    if _use_triton_flashmla is None:
        _use_triton_flashmla = (
            _is_sm120()
            and os.environ.get("SGLANG_SM120_TRITON_FLASHMLA", "0") == "1"
        )
    return _use_triton_flashmla


# SWA KV cache format constants
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_TILES = DIM_NOPE // TILE_SIZE  # 7
SCALE_PAD = 1
BYTES_NOPE_ROPE = DIM_NOPE + DIM_ROPE * 2  # 576
BYTES_SCALE = NUM_TILES + SCALE_PAD  # 8


def _is_sm120():
    global _SM120
    if _SM120 is None:
        _SM120 = torch.cuda.get_device_capability()[0] >= 12
    return _SM120


def _gather_and_dequant_kv_vectorized(k_cache, indices, topk_length=None):
    """Memory-efficient gather + dequantize KV tokens from paged cache.

    Processes tokens in chunks to limit peak memory usage.
    """
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    kv_dim = k_cache.shape[3]

    batch = indices.shape[0]
    topk = indices.shape[-1]

    idx_flat = indices.reshape(batch, -1)
    total_tokens = num_pages * page_size
    valid_mask = (idx_flat >= 0) & (idx_flat < total_tokens)
    idx_safe = idx_flat.clamp(0, total_tokens - 1)

    page_idx = idx_safe // page_size
    token_in_page = idx_safe % page_size

    raw_buf = k_cache.view(torch.uint8).reshape(-1)
    buf_len = raw_buf.shape[0]
    bytes_per_page = page_size * kv_dim
    page_offsets = page_idx * bytes_per_page

    # Pre-allocate output directly as bf16 to avoid intermediate fp32
    result = torch.zeros(batch, topk, DIM_NOPE + DIM_ROPE, dtype=torch.bfloat16, device=k_cache.device)

    # Compute all offsets once
    nope_starts = page_offsets + token_in_page * BYTES_NOPE_ROPE
    scale_section_offsets = page_offsets + page_size * BYTES_NOPE_ROPE
    scale_starts = scale_section_offsets + token_in_page * BYTES_SCALE

    # Gather NoPE FP8 + dequant in-place
    nope_offsets = nope_starts.unsqueeze(-1) + torch.arange(DIM_NOPE, device=raw_buf.device)
    nope_offsets_clamped = nope_offsets.clamp(0, buf_len - 1)
    nope_bytes = raw_buf[nope_offsets_clamped.reshape(-1)].reshape(batch, topk, DIM_NOPE)

    # Gather scales
    scale_offsets = scale_starts.unsqueeze(-1) + torch.arange(NUM_TILES, device=raw_buf.device)
    scale_offsets_clamped = scale_offsets.clamp(0, buf_len - 1)
    scale_bytes = raw_buf[scale_offsets_clamped.reshape(-1)].reshape(batch, topk, NUM_TILES)

    # Dequant: directly to bf16 to save memory (avoid large fp32 intermediate)
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)
    scale_fp32 = torch.pow(2.0, scale_bytes.float() - 127.0)
    # Dequant in tiled fashion and store directly to output
    nope_fp32 = nope_fp8.reshape(batch, topk, NUM_TILES, TILE_SIZE).float() * scale_fp32.unsqueeze(-1)
    result[:, :, :DIM_NOPE] = nope_fp32.reshape(batch, topk, DIM_NOPE).to(torch.bfloat16)

    # Free large intermediates
    del nope_bytes, nope_fp8, scale_bytes, scale_fp32, nope_fp32
    del nope_offsets, nope_offsets_clamped, scale_offsets, scale_offsets_clamped

    # Gather RoPE BF16 - much smaller (128 bytes per token)
    rope_starts = nope_starts + DIM_NOPE
    rope_byte_dim = DIM_ROPE * 2
    rope_offsets = rope_starts.unsqueeze(-1) + torch.arange(rope_byte_dim, device=raw_buf.device)
    rope_offsets_clamped = rope_offsets.clamp(0, buf_len - 1)
    rope_bytes = raw_buf[rope_offsets_clamped.reshape(-1)].reshape(batch, topk, rope_byte_dim)
    result[:, :, DIM_NOPE:] = rope_bytes.view(torch.bfloat16)

    # Zero invalid entries
    result = torch.where(valid_mask.unsqueeze(-1), result, torch.zeros_like(result))

    return result


def _flash_mla_with_kvcache_torch(
    q,
    k_cache,
    head_dim_v,
    tile_scheduler_metadata,
    softmax_scale=None,
    is_fp8_kvcache=True,
    indices=None,
    topk_length=None,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    **kwargs,
):
    """Pure PyTorch fallback for flash_mla sparse decode on SM120.

    When SGLANG_SM120_TRITON_FLASHMLA=1, uses the tiled Triton kernel
    which fuses gather + dequant + QK + softmax + V into a single kernel.
    Falls back to the vectorized PyTorch path otherwise.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    batch, seq_q, num_heads_q, head_dim = q.shape
    device = q.device

    if indices is None:
        raise NotImplementedError(
            "Dense decode path not implemented for SM120 fallback"
        )

    # Fast path: Triton tiled sparse decode (fuses all ops into one kernel)
    if _should_use_triton_flashmla():
        from sglang.srt.layers.attention.flash_mla_sm120_triton import (
            flash_mla_sparse_decode_triton,
        )

        return flash_mla_sparse_decode_triton(
            q=q,
            k_cache=k_cache,
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
        )

    # Slow path: vectorized PyTorch fallback
    if _is_sm120() and _should_use_triton_gather():
        from sglang.srt.layers.attention.fused_kv_gather_triton import (
            fused_gather_dequant,
        )

        k_all = fused_gather_dequant(k_cache, indices)
    else:
        k_all = _gather_and_dequant_kv_vectorized(k_cache, indices, topk_length)

    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        extra_k = _gather_and_dequant_kv_vectorized(
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length
        )
        k_all = torch.cat([k_all, extra_k], dim=1)

    num_kv_tokens = k_all.shape[1]

    v = k_all  # (batch, num_kv, 512)

    q_full = q.squeeze(1)  # (batch, num_heads, 512)
    scores = torch.einsum("bhd,bkd->bhk", q_full, k_all) * softmax_scale
    scores_4d = scores.unsqueeze(1)  # (batch, 1, num_heads, num_kv)

    if topk_length is not None:
        total_kv = num_kv_tokens
        primary_topk = indices.shape[-1]
        arange = torch.arange(total_kv, device=device)

        primary_valid = arange[:primary_topk].unsqueeze(0) < topk_length.unsqueeze(1)
        if extra_k_cache is not None and extra_topk_length is not None:
            extra_valid = (arange[primary_topk:] - primary_topk).unsqueeze(
                0
            ) < extra_topk_length.unsqueeze(1)
            valid_mask = torch.cat([primary_valid, extra_valid], dim=1)
        else:
            valid_mask = primary_valid

        scores_4d = scores_4d.masked_fill(
            ~valid_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

    attn_weights = torch.softmax(scores_4d.float(), dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    attn_3d = attn_weights.squeeze(1).reshape(batch, num_heads_q, num_kv_tokens)

    o = torch.einsum("bhk,bkd->bhd", attn_3d, v.to(attn_3d.dtype))
    o = o.unsqueeze(1)  # (batch, 1, num_heads, 512)

    lse_out = torch.logsumexp(scores_4d.float(), dim=-1).transpose(-1, -2)

    return (o.to(torch.bfloat16), lse_out)


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend == "kernel":
        if _is_sm120():
            return _flash_mla_with_kvcache_torch(**kwargs)
        else:
            import flash_mla

            return flash_mla.flash_mla_with_kvcache(**kwargs)
    raise ValueError(f"unsupported backend {backend!r}")
