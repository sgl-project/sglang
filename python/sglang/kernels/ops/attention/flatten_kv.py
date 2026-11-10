"""
FlattenKV: gather scattered paged KV into contiguous buffer for SM70 prefill.

On V100, the scatter-gather of KV through page table lookups during attention
tile iteration is prohibitively expensive. This kernel pre-assembles a contiguous
[total_seq_tokens, num_heads, head_dim] buffer so the attention kernel reads
sequential memory instead of scattered blocks.

Implementation: simple gather using req_to_token[seq, pos] → physical index,
concatenated with new KV tokens. Uses advanced indexing (efficient on modern GPUs).

For production fused performance, this can be replaced by a Triton/CUDA kernel.
"""

import torch
import torch.nn.functional as F


def flatten_kv(
    req_to_token: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Flatten paged KV to contiguous buffer for SM70 prefill.

    Args:
        req_to_token: [max_bs+1, max_context_len] — logical→physical mapping
                      (req_to_token[seq_idx, token_pos] = physical KV cache index)
        k_paged: [pool_size, num_heads, head_dim] — paged KV cache
        v_paged: same layout
        k_new: [num_extend_tokens, num_heads, head_dim] — new K after set_kv_buffer
        v_new: same layout
        seq_lens: [num_seqs] — total sequence lengths per sequence
        extend_prefix_lens: [num_seqs] — cached prefix length per sequence
        device: torch device

    Returns:
        (flat_k, flat_v, flat_cu_seqlens, seq_lens) or (None, None, None, None)
        flat_k, flat_v: [total_seq_tokens, num_heads, head_dim] contiguous
        flat_cu_seqlens: [num_seqs+1] cumulative positions for ragged wrapper
        seq_lens: same as input, for ragged wrapper

        Returns None tuple when:
        - No prefix tokens exist (all new)
        - No sequences to flatten
        """
    num_seqs = len(seq_lens)
    if num_seqs <= 0:
        return None, None, None, None

    num_heads = k_new.shape[1]
    head_dim = k_new.shape[2]
    total_tokens = int(seq_lens.sum().item())
    prefix_len_sum = int(extend_prefix_lens.sum().item())

    # No prefix = nothing to flatten (all tokens are new and already contiguous in k_new)
    if prefix_len_sum == 0:
        return None, None, None, None

    # Allocate contiguous output buffer
    flat_k = torch.zeros(
        total_tokens, num_heads, head_dim,
        dtype=k_paged.dtype, device=device
    )
    flat_v = torch.zeros(
        total_tokens, num_heads, head_dim,
        dtype=k_paged.dtype, device=device
    )

    # Compute cu_seqlens for ragged wrapper
    flat_cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    flat_cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

    # Build flat buffer per-sequence:
    # For each seq: prefix KV (gathered from paged) + new KV (from k_new)
    for seq_idx in range(num_seqs):
        prefix_len = int(extend_prefix_lens[seq_idx])
        seq_len = int(seq_lens[seq_idx])
        flat_start = int(flat_cu_seqlens[seq_idx])
        flat_end = int(flat_cu_seqlens[seq_idx + 1])

        # Gather prefix KV from paged cache using req_to_token
        if prefix_len > 0:
            phys_idx = req_to_token[seq_idx : seq_idx + 1, :prefix_len].flatten()
            # Indexing: k_paged[phys_idx] gathers all prefix tokens into [prefix_len, num_heads, head_dim]
            flat_k[flat_start : flat_start + prefix_len] = k_paged[phys_idx]
            flat_v[flat_start : flat_start + prefix_len] = v_paged[phys_idx]

        # Place new KV tokens (already contiguous after set_kv_buffer)
        n_new = seq_len - prefix_len
        if n_new > 0:
            # Count total new tokens up to this sequence for k_new indexing
            new_start = 0
            for s in range(seq_idx):
                new_start += int(seq_lens[s].item()) - int(extend_prefix_lens[s].item())
            flat_k[flat_start + prefix_len : flat_end] = k_new[new_start : new_start + n_new]
            flat_v[flat_start + prefix_len : flat_end] = v_new[new_start : new_start + n_new]

    return flat_k, flat_v, flat_cu_seqlens, seq_lens
