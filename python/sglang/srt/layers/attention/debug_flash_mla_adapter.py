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

import torch

_SM120 = None
_use_triton_gather = None


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
    """Vectorized gather + dequantize KV tokens from paged cache.

    Single-pass batch gather: fewer kernel launches than tile-by-tile.
    Skips .contiguous() when cache is already contiguous.
    """
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]  # 256
    kv_dim = k_cache.shape[3]  # 584

    batch = indices.shape[0]
    topk = indices.shape[-1]

    idx_flat = indices.reshape(batch, -1)  # (batch, topk)

    # Clamp invalid indices
    total_tokens = num_pages * page_size
    valid_mask = (idx_flat >= 0) & (idx_flat < total_tokens)
    idx_safe = idx_flat.clamp(0, total_tokens - 1)

    # Map flat index to (page_idx, token_in_page)
    page_idx = idx_safe // page_size  # (batch, topk)
    token_in_page = idx_safe % page_size  # (batch, topk)

    # Assume contiguous for CUDA graph compatibility (always true in decode)
    raw_buf = k_cache.view(torch.uint8).reshape(-1)

    buf_len = raw_buf.shape[0]

    # Byte offsets for each page (int32 arithmetic, implicit int64 on indexing)
    bytes_per_page = page_size * kv_dim  # 256 * 584 = 149504
    page_offsets = page_idx * bytes_per_page  # (batch, topk) int32

    # === Section A: NoPE FP8 + RoPE BF16 ===
    nope_starts = page_offsets + token_in_page * BYTES_NOPE_ROPE  # (batch, topk) int32
    rope_starts = nope_starts + DIM_NOPE  # (batch, topk)

    # === Section B: UE8M0 Scales ===
    scale_section_offsets = (
        page_offsets + page_size * BYTES_NOPE_ROPE
    )  # page_size * 576
    scale_starts = scale_section_offsets + token_in_page * BYTES_SCALE  # (batch, topk)

    # Gather all NoPE bytes: (batch, topk, 448) uint8 - single batched gather
    nope_offsets = nope_starts.unsqueeze(-1) + torch.arange(
        DIM_NOPE, device=raw_buf.device
    )
    nope_offsets_clamped = nope_offsets.clamp(0, buf_len - 1)
    nope_bytes = raw_buf[nope_offsets_clamped.reshape(-1)].reshape(
        batch, topk, DIM_NOPE
    )
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn).to(torch.float32)

    # Gather all RoPE bytes: (batch, topk, 128) uint8 -> (batch, topk, 64) bf16
    rope_byte_dim = DIM_ROPE * 2  # 128 bytes = 64 bf16 values
    rope_offsets = rope_starts.unsqueeze(-1) + torch.arange(
        rope_byte_dim, device=raw_buf.device
    )
    rope_offsets_clamped = rope_offsets.clamp(0, buf_len - 1)
    rope_bytes = raw_buf[rope_offsets_clamped.reshape(-1)].reshape(
        batch, topk, rope_byte_dim
    )
    rope_bf16 = rope_bytes.view(torch.bfloat16)

    # Gather all scale bytes: (batch, topk, 7) uint8 - single batched gather
    scale_offsets = scale_starts.unsqueeze(-1) + torch.arange(
        NUM_TILES, device=raw_buf.device
    )
    scale_offsets_clamped = scale_offsets.clamp(0, buf_len - 1)
    scale_bytes = raw_buf[scale_offsets_clamped.reshape(-1)].reshape(
        batch, topk, NUM_TILES
    )

    # Dequantize UE8M0 scales: scale = pow(2, uint8_value - 127)
    scale_fp32 = torch.pow(
        2.0, scale_bytes.to(torch.float32) - 127.0
    )  # (batch, topk, 7)

    # Dequantize NoPE: fp32_value = fp8_value * scale - single batched multiply
    nope_fp32 = nope_fp8.reshape(
        batch, topk, NUM_TILES, TILE_SIZE
    ) * scale_fp32.unsqueeze(-1)
    nope_bf16 = nope_fp32.reshape(batch, topk, DIM_NOPE).to(torch.bfloat16)

    # Concatenate NoPE + RoPE -> (batch, topk, 512)
    result = torch.cat([nope_bf16, rope_bf16], dim=-1)

    # Zero out invalid entries
    # Zero invalid entries using torch.where (CUDA graph compatible)
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

    Optimized: single combined Q@K^T bmm instead of separate NoPE/RoPE einsums,
    reducing kernel launches from 3 to 1 for score computation.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    batch, seq_q, num_heads_q, head_dim = q.shape
    device = q.device

    if indices is None:
        raise NotImplementedError(
            "Dense decode path not implemented for SM120 fallback"
        )

    # Gather and dequantize primary KV tokens
    # Use Triton fused kernel on SM120 to reduce kernel launch overhead
    if _is_sm120() and _should_use_triton_gather():
        from sglang.srt.layers.attention.fused_kv_gather_triton import (
            fused_gather_dequant,
        )

        k_all = fused_gather_dequant(k_cache, indices)
    else:
        k_all = _gather_and_dequant_kv_vectorized(k_cache, indices, topk_length)

    # Gather extra KV tokens if present
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        if _is_sm120() and _should_use_triton_gather():
            from sglang.srt.layers.attention.fused_kv_gather_triton import (
                fused_gather_dequant,
            )

            extra_k = fused_gather_dequant(extra_k_cache, extra_indices_in_kvcache)
        else:
            extra_k = _gather_and_dequant_kv_vectorized(
                extra_k_cache, extra_indices_in_kvcache, extra_topk_length
            )
        k_all = torch.cat([k_all, extra_k], dim=1)

    num_kv_tokens = k_all.shape[1]

    # MLA: V = K (full 512-dim)
    v = k_all  # (batch, num_kv, 512)

    # Fused attention scores: single full-dim einsum (saves 1 kernel launch per layer)
    q_full = q.squeeze(1)  # (batch, num_heads, 512)
    scores = torch.einsum("bhd,bkd->bhk", q_full, k_all) * softmax_scale
    scores_4d = scores.unsqueeze(1)  # (batch, 1, num_heads, num_kv)

    # Causal mask via topk_length
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

    # Softmax + output
    attn_weights = torch.softmax(scores_4d.float(), dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)

    # attn_weights: (batch, 1, num_heads, num_kv) -> (batch, num_heads, num_kv)
    attn_3d = attn_weights.squeeze(1).reshape(batch, num_heads_q, num_kv_tokens)

    # Output: attn_weights @ V
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
