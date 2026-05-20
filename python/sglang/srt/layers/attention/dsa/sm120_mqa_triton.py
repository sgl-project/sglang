"""SM120-optimized MQA logits — CUDA graph compatible.

Replaces the PyTorch fallback in sm120_mqa_fallback.py with an optimized
implementation that precomputes the head-weighted query vector before
scanning the KV cache, reducing per-position work from O(n_heads) to O(1).

Key insight: logit[s] = sum_h(w[h] * dot(q[h], kv[s]))
           = dot(sum_h(w[h] * q[h]), kv[s])
           = dot(wq, kv[s])

CUDA graph compatibility:
- No .item() calls — all computation stays on GPU tensors
- No per-batch Python loops — vectorized with torch.bmm
- Fixed tensor shapes derived from known parameters (max_seq_len, num_k)

Target: RTX PRO 6000 (SM120, 188 SMs, 99KB SMEM, ~1.5 TB/s GDDR7)
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _dequant_fp8_with_scale_suffix(
    data_fp8: torch.Tensor,
    head_dim_qk: int,
) -> torch.Tensor:
    """Dequantize FP8 tensor with appended float32 scale suffix."""
    data_bytes = data_fp8[..., :head_dim_qk]
    scale_bytes = data_fp8[..., head_dim_qk:]
    scale = scale_bytes.contiguous().view(torch.float32)
    return data_bytes.to(torch.float32) * scale


def compute_paged_mqa_schedule_metadata(
    seqlens: torch.Tensor,
    block_size: int,
    num_sms: int,
) -> None:
    """SM120 fallback: scheduling is handled internally, return None."""
    return None


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
    """CUDA-graph-compatible paged MQA logits for SM120.

    Key optimizations vs fallback:
    1. Precompute wq = sum_h(w[h] * dequant(q[h])) — eliminates per-position head loop
    2. Batched matmul across all batch elements — no per-batch Python loop
    3. No .item() calls — all shapes derived from known parameters
    """
    batch, next_n, n_heads, hd_with_sf = q_fp8.shape
    hd = hd_with_sf - 4
    block_kv = kv_cache_fp8.shape[1]
    device = q_fp8.device

    seqlens_flat = seqlens.view(-1).to(torch.int64)

    # Dequant Q: [batch, next_n, n_heads, hd]
    q_f32 = _dequant_fp8_with_scale_suffix(q_fp8, hd)

    # Precompute wq = sum_h(w[b,h] * q[b,t,h,:]) → [batch, next_n, hd]
    w = weights.view(batch, 1, n_heads, 1)
    wq = (q_f32 * w).sum(dim=2)  # [batch, next_n, hd]

    # Batch-dequant all KV blocks: [num_blocks, block_kv, hd]
    kv_data = kv_cache_fp8[..., :hd].squeeze(2)
    kv_scale_raw = kv_cache_fp8[..., hd:].squeeze(2)
    kv_scale = kv_scale_raw.contiguous().view(torch.float32)
    kv_f32 = kv_data.float() * kv_scale  # [num_blocks_total, block_kv, hd]

    # ── Vectorized batch gather (no per-batch loop, no .item()) ──
    max_blocks = (max_seq_len + block_kv - 1) // block_kv
    # Gather block IDs for all batches: [batch, max_blocks]
    block_ids = block_tables[:, :max_blocks]

    # Gather KV for all batches: [batch, max_blocks, block_kv, hd]
    kv_batched = kv_f32[block_ids]
    max_padded = max_blocks * block_kv
    kv_flat = kv_batched.reshape(batch, max_padded, hd)

    # Batched matmul: [batch, next_n, hd] @ [batch, hd, max_padded]
    logits_batched = torch.bmm(
        wq, kv_flat.transpose(1, 2)
    )  # [batch, next_n, max_padded]

    # Create validity mask: [batch, max_padded]
    positions = torch.arange(max_padded, device=device)
    valid = positions.unsqueeze(0) < seqlens_flat.unsqueeze(1)  # [batch, max_padded]

    # Apply mask (broadcast over next_n)
    logits_batched = logits_batched.masked_fill(~valid.unsqueeze(1), float("-inf"))

    # Write to output: [batch * next_n, max_seq_len]
    out_width = min(max_padded, max_seq_len)
    out = torch.full(
        (batch * next_n, max_seq_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    out[:, :out_width] = logits_batched[:, :, :out_width].reshape(
        batch * next_n, out_width
    )

    return out


def sm120_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """CUDA-graph-compatible ragged MQA logits for SM120.

    Key optimization: precompute wq = sum_h(w[h] * q[h]), then single matmul.
    No .item() calls — uses num_k for output width.
    """
    num_q, n_heads, hd_with_sf = q_fp8.shape
    hd = hd_with_sf - 4
    device = q_fp8.device

    k_data, k_scale = kv_fp8
    num_k = k_data.shape[0]

    # Use num_k as output width — avoids ke.max().item() GPU-CPU sync
    out_width = num_k

    out = torch.full(
        (num_q, out_width),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    if num_q == 0 or num_k == 0:
        return out

    # Dequant Q and precompute weighted query
    q_f32 = _dequant_fp8_with_scale_suffix(q_fp8, hd)
    w = weights.unsqueeze(-1)
    wq = (q_f32 * w).sum(dim=1)  # [num_q, hd]

    # Dequant KV
    if k_data.shape[-1] == hd_with_sf:
        k_f32 = _dequant_fp8_with_scale_suffix(k_data.unsqueeze(-2), hd).squeeze(-2)
    else:
        k_f32 = k_data.float()
        if k_scale.dim() == 1:
            k_f32 = k_f32 * k_scale.unsqueeze(-1)
        else:
            k_f32 = k_f32 * k_scale

    # Single matmul: [num_q, hd] @ [hd, num_k] → [num_q, num_k]
    logits_all = wq @ k_f32.T

    # Apply ragged [ks, ke) masking
    k_indices = torch.arange(out_width, device=device).unsqueeze(0)
    mask = (k_indices >= ks.unsqueeze(1)) & (k_indices < ke.unsqueeze(1))
    out[:, :num_k] = torch.where(mask[:, :num_k], logits_all, out[:, :num_k])

    return out
