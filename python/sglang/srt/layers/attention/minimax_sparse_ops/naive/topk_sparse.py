# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch


def naive_flash_decode_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, head_dim]
    kv_cache: torch.Tensor,  # [max_slots, 2, max_len, num_kv_heads, head_dim]
    seq_lens: torch.Tensor,  # [batch_size, ]
    slot_ids: torch.Tensor,  # [batch_size, ]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Naive implementation of sparse attention with GQA group sharing.
    Each GQA group (all q heads sharing the same kv head) uses the same topk_idx.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    original_dtype = q.dtype
    batch_size, num_q_heads, head_dim = q.shape
    max_slots, _, max_kv_len, num_kv_heads, _ = kv_cache.shape
    gqa_group_size = num_q_heads // num_kv_heads
    # Output tensor
    o = torch.zeros(batch_size, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        sid = slot_ids[b].item() % max_slots

        for kh in range(num_kv_heads):
            # Get topk indices for this (kv_head, batch)
            block_indices = topk_idx[kh, b, :].tolist()

            # Collect selected K and V blocks
            selected_k_blocks = []
            selected_v_blocks = []
            for block_idx in block_indices:
                if block_idx < 0:
                    continue  # Invalid index
                start = block_idx * block_size
                end = min(start + block_size, seq_len)
                if start >= seq_len:
                    continue
                # K: [block_len, head_dim]
                k_block = kv_cache[sid, 0, start:end, kh, :]
                # V: [block_len, head_dim]
                v_block = kv_cache[sid, 1, start:end, kh, :]
                selected_k_blocks.append(k_block)
                selected_v_blocks.append(v_block)

            if len(selected_k_blocks) == 0:
                continue

            # Concatenate selected blocks: [total_selected_len, head_dim]
            k_selected = torch.cat(selected_k_blocks, dim=0)
            v_selected = torch.cat(selected_v_blocks, dim=0)

            # Compute attention for all q heads in this GQA group
            for g in range(gqa_group_size):
                qh = kh * gqa_group_size + g
                # q_vec: [head_dim]
                q_vec = q[b, qh, :].float()

                # Compute attention scores: [total_selected_len]
                scores = torch.matmul(q_vec, k_selected.float().T) * sm_scale

                # Add sink to softmax normalization if present
                if sink is not None:
                    sink_vec = sink[qh, :].float()
                    qsink = torch.dot(q_vec, sink_vec) * sm_scale
                    # Concatenate sink score with regular scores for softmax
                    scores_with_sink = torch.cat([qsink.unsqueeze(0), scores], dim=0)
                    attn_weights_with_sink = torch.softmax(scores_with_sink, dim=-1)
                    # Remove sink weight (only used for normalization)
                    attn_weights = attn_weights_with_sink[1:]
                else:
                    attn_weights = torch.softmax(scores, dim=-1)

                # Compute output: [head_dim]
                o[b, qh, :] = torch.matmul(attn_weights.to(original_dtype), v_selected)

    return o
