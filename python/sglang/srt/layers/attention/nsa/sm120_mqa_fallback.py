"""
SM120 fallback kernels for DeepGEMM FP8 MQA logits operations.

On SM120 (RTX 5090, RTX PRO 6000, DGX Spark), DeepGEMM's fp8_paged_mqa_logits
and fp8_mqa_logits crash with 'Unsupported architecture'. This module provides
PyTorch-native fallback implementations that match the DeepGEMM API contract.

Reference: vLLM PR#40991 (Triton sparse MLA fallback approach for SM120)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def compute_paged_mqa_schedule_metadata(
    seqlens: torch.Tensor,
    block_size: int,
    num_sms: int,
) -> None:
    """SM120 fallback: scheduling is handled internally, return None."""
    return None


def _dequant_fp8_with_scale_suffix(
    data_fp8: torch.Tensor, head_dim_qk: int
) -> torch.Tensor:
    """
    Dequantize FP8 tensor that has per-row scale factors appended.

    DeepGEMM packs KV cache as [data_fp8 (head_dim_qk bytes) | scale (4 bytes)]
    in a tensor of shape [..., head_dim_with_sf] where head_dim_with_sf = head_dim_qk + 4.
    The scale is stored as a float32 value in the last 4 bytes.
    """
    # Split data and scale
    data_bytes = data_fp8[..., :head_dim_qk]
    # Scale is stored in the last 4 bytes, reinterpret as float32
    scale_bytes = data_fp8[..., head_dim_qk:]
    scale = scale_bytes.contiguous().view(torch.float32)  # [..., 1]

    # Dequantize: cast FP8 to float32, multiply by scale
    data_f32 = data_bytes.to(torch.float32) * scale
    return data_f32


def sm120_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """
    SM120 fallback for deep_gemm.fp8_paged_mqa_logits().

    Computes weighted multi-head dot-product logits over paged KV cache.

    Args:
        q_fp8: [batch, next_n, n_heads, head_dim_with_sf] FP8 queries with appended scale
        kv_cache_fp8: [num_blocks, block_kv, 1, head_dim_with_sf] FP8 paged KV cache
        weights: [batch, n_heads] float32 head weights
        seqlens: [batch, 1] or [batch] int32 sequence lengths
        block_tables: [batch, max_blocks] int32 block table indices
        schedule_metadata: ignored on SM120 (None)
        max_seq_len: maximum sequence length for output
        clean_logits: if True, fill unused positions with -inf

    Returns:
        logits: [batch * next_n, max_seq_len] float32
    """
    batch, next_n, n_heads, head_dim_with_sf = q_fp8.shape
    head_dim_qk = head_dim_with_sf - 4  # 128 typically
    block_kv = kv_cache_fp8.shape[1]  # typically 64
    device = q_fp8.device

    # Flatten seqlens
    if seqlens.dim() == 2:
        seqlens = seqlens.squeeze(-1)

    # Output logits
    out = torch.full(
        (batch * next_n, max_seq_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    # Dequantize queries: [batch, next_n, n_heads, head_dim_qk]
    q_f32 = _dequant_fp8_with_scale_suffix(q_fp8, head_dim_qk)

    for b in range(batch):
        seq_len = seqlens[b].item()
        if seq_len <= 0:
            continue

        num_blocks_needed = (seq_len + block_kv - 1) // block_kv

        # Gather KV blocks for this batch element
        block_ids = block_tables[b, :num_blocks_needed]
        # [num_blocks_needed, block_kv, 1, head_dim_with_sf]
        kv_blocks = kv_cache_fp8[block_ids]
        # Flatten to [num_blocks_needed * block_kv, head_dim_with_sf]
        kv_flat = kv_blocks.view(-1, head_dim_with_sf)
        # Trim to actual sequence length
        kv_flat = kv_flat[:seq_len]

        # Dequantize KV: [seq_len, head_dim_qk]
        k_f32 = _dequant_fp8_with_scale_suffix(kv_flat.unsqueeze(-2), head_dim_qk)
        k_f32 = k_f32.squeeze(-2)  # [seq_len, head_dim_qk]

        # Vectorized over next_n:
        # q_b: [next_n, n_heads, head_dim_qk]
        q_b = q_f32[b]
        # dots: [next_n, n_heads, seq_len]
        dots = torch.einsum("tnd,sd->tns", q_b, k_f32)
        # Apply head weights: [n_heads] -> weighted sum -> [next_n, seq_len]
        w = weights[b]  # [n_heads]
        logits_b = torch.einsum("tns,n->ts", dots, w)  # [next_n, seq_len]
        out_start = b * next_n
        out[out_start:out_start + next_n, :seq_len] = logits_b

    return out


def sm120_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """
    SM120 fallback for deep_gemm.fp8_mqa_logits() (contiguous/ragged variant).

    Computes weighted multi-head dot-product logits over contiguous KV.

    Args:
        q_fp8: [num_q, n_heads, head_dim_with_sf] FP8 queries with appended scale
        kv_fp8: tuple of (k_data_fp8 [num_k, head_dim_with_sf], k_scale [num_k]) or
                (k_data_fp8 [num_k, D], k_scale [num_k, scale_dim])
        weights: [num_q, n_heads] float32 head weights
        ks: [num_q] int32 start indices into KV
        ke: [num_q] int32 end indices into KV

    Returns:
        logits: [num_q, num_k] float32 where num_k = max(ke) - min(ks) (or ke.max())
    """
    num_q, n_heads, head_dim_with_sf = q_fp8.shape
    head_dim_qk = head_dim_with_sf - 4
    device = q_fp8.device

    k_data, k_scale = kv_fp8
    num_k = k_data.shape[0]

    # Determine output width
    k_max = ke.max().item() if ke.numel() > 0 else 0
    out_width = max(k_max, num_k)

    # Output logits
    out = torch.full(
        (num_q, out_width),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    if num_q == 0 or num_k == 0:
        return out

    # Dequantize queries: [num_q, n_heads, head_dim_qk]
    q_f32 = _dequant_fp8_with_scale_suffix(q_fp8, head_dim_qk)

    # Dequantize KV keys
    if k_data.shape[-1] == head_dim_with_sf:
        # Keys have appended scale suffix
        k_f32 = _dequant_fp8_with_scale_suffix(k_data.unsqueeze(-2), head_dim_qk)
        k_f32 = k_f32.squeeze(-2)  # [num_k, head_dim_qk]
    else:
        # Keys and scales are separate
        k_f32 = k_data.to(torch.float32)
        if k_scale.dim() == 1:
            k_f32 = k_f32 * k_scale.unsqueeze(-1)
        else:
            k_f32 = k_f32 * k_scale

    # Vectorized: compute all dot products at once
    # q_f32: [num_q, n_heads, head_dim_qk], k_f32: [num_k, head_dim_qk]
    # dots: [num_q, n_heads, num_k]
    dots = torch.einsum("qhd,kd->qhk", q_f32, k_f32)

    # Apply head weights: [num_q, n_heads] -> [num_q, n_heads, 1]
    w = weights.unsqueeze(-1)
    # Weighted sum across heads: [num_q, num_k]
    logits_all = (dots * w).sum(dim=1)

    # Mask to [ks, ke) ranges
    k_indices = torch.arange(out_width, device=device).unsqueeze(0)  # [1, out_width]
    ks_expanded = ks.unsqueeze(1)  # [num_q, 1]
    ke_expanded = ke.unsqueeze(1)  # [num_q, 1]
    mask = (k_indices >= ks_expanded) & (k_indices < ke_expanded)  # [num_q, out_width]

    # Place logits into output at valid positions
    # logits_all is [num_q, num_k], but output is [num_q, out_width]
    out[:, :num_k] = torch.where(mask[:, :num_k], logits_all, out[:, :num_k])

    return out
