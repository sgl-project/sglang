import math
from typing import Optional

import torch
from einops import rearrange, repeat


def construct_streaming_mask(
    seqlen_q: int,
    seqlen_k: int,
    sink_size: int,
    local_size: int,
    is_causal: bool,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Construct attention mask for Streaming Attention.

    This mask combines "attention sink" and "local sliding window".
    A query token `i` can attend to:
    1. The first `sink_size` tokens at the beginning of the sequence.
    2. `local_size` tokens within its local window (including itself).

    Args:
        seqlen_q (int): Length of the query sequence.
        seqlen_k (int): Length of the key sequence.
        sink_size (int): Size of the attention sink region.
        local_size (int): Size of the local sliding window.
        is_causal (bool): Whether to apply causal constraint.
        query_padding_mask (Optional[torch.Tensor]): Padding mask for query sequence, shape (seqlen_q,).
                                                     True indicates valid token, False indicates padding.
        key_padding_mask (Optional[torch.Tensor]): Padding mask for key sequence, shape (seqlen_k,).
                                                   True indicates valid token, False indicates padding.
        device (torch.device): Device where the tensor resides.

    Returns:
        torch.Tensor: A boolean attention mask tensor with shape (seqlen_q, seqlen_k).
                      `True` indicates the position should be masked (ignored), `False` indicates retained.
    """
    assert sink_size >= 0, "sink_size must be greater than or equal to 0"
    assert local_size >= 1, "local_size must be greater than 0"

    # Create row and column indices to build coordinate grid for attention matrix
    # row_idx represents query token position (i)
    # rearrange changes shape from (s,) to (s, 1) for broadcasting with col_idx
    row_idx = rearrange(torch.arange(seqlen_q, device=device), "s -> s 1")

    # col_idx represents key token position (j)
    col_idx = torch.arange(seqlen_k, device=device)

    # Only perform complex mask computation when causal constraint is needed
    if is_causal:
        # --- Handle cases with padding ---
        # If padding masks are provided, we need to compute the actual length of each sequence,
        # because causal relationships and window positions are relative to actual tokens, not fixed sequence length.
        if query_padding_mask is not None or key_padding_mask is not None:
            # sk is the actual length of the key sequence. If no mask is provided, use seqlen_k.
            # Otherwise, get actual length by summing the mask (assuming real tokens are 1, padding is 0).
            # Note: padding_mask is now a mask for a single sequence with shape (seqlen,), not batch-level
            sk = seqlen_k if key_padding_mask is None else key_padding_mask.sum().item()
            # sq is the actual length of the query sequence, same logic as above.
            sq = (
                seqlen_q
                if query_padding_mask is None
                else query_padding_mask.sum().item()
            )
        # Handle simple case without padding (seqlen_q, seqlen_k)
        else:
            sk = seqlen_k
            sq = seqlen_q

        if query_padding_mask is not None or key_padding_mask is not None:
            # For cases with padding, need to adjust causal relationship and window boundaries
            # because actual sequence may be shorter than padded length
            beyond_causal = col_idx > torch.minimum(
                row_idx + sk - sq, torch.tensor(sk, device=device)
            )
            outside_window = torch.logical_and(
                col_idx < row_idx + sk - sq - (local_size - 1), col_idx >= sink_size
            )
        else:
            # Simplified logic without padding
            # Condition 1: Causal mask
            # Key position `j` (col_idx) cannot be greater than query position `i` (row_idx).
            # This creates an upper triangular matrix (True above diagonal).
            # beyond_causal has shape (seqlen_q, seqlen_k)
            # because row_idx has shape (seqlen_q, 1), col_idx has shape (seqlen_k,)
            # Through broadcasting, the comparison produces a boolean tensor of shape (seqlen_q, seqlen_k)
            beyond_causal = col_idx > row_idx

            # Condition 2: Outside window mask
            # `row_idx - (local_size - 1)` is the left boundary of the sliding window.
            # For example, for query i=7, local_size=3, window is [5, 6, 7]. Left boundary is 7-(3-1)=5.
            # Any token with col_idx < 5 is outside the window.
            # `col_idx >= sink_size` exempts tokens in the sink region.
            outside_window = torch.logical_and(
                col_idx < row_idx - (local_size - 1), col_idx >= sink_size
            )
        mask = torch.logical_or(beyond_causal, outside_window)

    # If not causal mode (e.g., encoder like BERT), don't apply any mask.
    # Create an all-False mask, allowing all tokens to attend to each other.
    else:
        mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    return mask


def block_streaming_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_mask_type: torch.Tensor,
    sink_size: int,
    local_size: int,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = True,
    dropout_mask: Optional[torch.Tensor] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    return_attn_probs: bool = False,
):
    """
    Implement block-based attention mechanism supporting mixed modes (dense, streaming).

    Args:
        q, k, v (torch.Tensor): Input query, key, value tensors. They are concatenated results of all sequences in the batch.
                                Shape is (total_tokens, num_heads, head_dim).
        cu_seqlens_q, cu_seqlens_k (torch.Tensor): Cumulative sequence lengths.
                                                  For example, for a batch with lengths [L1, L2], cu_seqlens is [0, L1, L1+L2].
                                                  Used to slice each sequence from q, k, v.
        max_seqlen_q, max_seqlen_k (int): Maximum query/key sequence length in the batch.
        head_mask_type (torch.Tensor): A tensor of shape (num_heads,) determining attention type for each head.
                                       0: Dense Attention
                                       <0: Streaming Attention
                                       >0: Block Sparse Attention (not implemented here)
        sink_size (int): Size of the attention sink region.
        local_size (int): Size of the local sliding window.
        p_dropout (float): Dropout probability.
        softmax_scale (Optional[float]): Scaling factor for softmax.
        is_causal (bool): Whether to apply causal mask.
        ... (other parameters)
    """
    device = q.device
    total_q, num_heads, head_dim = q.shape
    _, num_heads_k, _ = k.shape

    # batch_size can be inferred from the length of cu_seqlens
    batch_size = cu_seqlens_q.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    attn_weight_lists = [] if return_attn_probs else None

    # --- 2. Handle GQA/MQA (Grouped Query Attention/Multi-Query Attention) ---
    # GQA/MQA is an optimization where multiple query heads share the same set of key/value heads to reduce KV cache size
    if num_heads_k != num_heads:
        # Ensure the number of query heads is an integer multiple of key/value heads
        assert num_heads % num_heads_k == 0
        # Use einops.repeat to replicate K and V heads to match Q heads for subsequent computation
        k = repeat(k, "t h d -> t (h g) d", g=num_heads // num_heads_k)
        v = repeat(v, "t h d -> t (h g) d", g=num_heads // num_heads_k)

    output = torch.zeros(total_q, num_heads, head_dim, device=device, dtype=q.dtype)

    # --- 3. Main loop: process each sequence in the batch ---
    for batch_idx in range(batch_size):
        # Use cumulative lengths to determine start and end positions of current sequence in concatenated tensor
        q_start, q_end = (
            cu_seqlens_q[batch_idx].item(),
            cu_seqlens_q[batch_idx + 1].item(),
        )
        k_start, k_end = (
            cu_seqlens_k[batch_idx].item(),
            cu_seqlens_k[batch_idx + 1].item(),
        )

        # Slice current sequence's q, k, v from large tensor
        q_batch = q[q_start:q_end]  # (seqlen_q, num_heads, head_dim)
        k_batch = k[k_start:k_end]
        v_batch = v[k_start:k_end]

        seqlen_q, seqlen_k = (
            q_batch.shape[0],
            k_batch.shape[0],
        )  # query sequence length, key sequence length

        # --- 4. Compute attention scores ---
        # Use einsum to efficiently compute dot product of Q and K, obtaining raw attention scores
        # "qhd,khd->hqk" means:
        # qhd: (seqlen_q, num_heads, head_dim)
        # khd: (seqlen_k, num_heads, head_dim)
        # hqk: output (num_heads, seqlen_q, seqlen_k)
        scores = torch.einsum(
            "qhd,khd->hqk", q_batch * softmax_scale, k_batch
        )  # (seqlen_q, seqlen_k)

        # If there's a key padding mask, set scores at padding positions to negative infinity
        # This way, after softmax, probabilities at these positions become 0
        if key_padding_mask is not None:
            key_mask = key_padding_mask[batch_idx, :seqlen_k].to(device)
            scores = scores.masked_fill(~key_mask[None, None, :], float("-inf"))

        for head_idx in range(num_heads):
            mask_type = head_mask_type[head_idx].item()

            if mask_type == 0:
                # Dense Attention
                if is_causal:
                    causal_mask = torch.triu(
                        torch.ones(
                            (seqlen_q, seqlen_k), dtype=torch.bool, device=device
                        ),
                        diagonal=1,
                    )
                    scores[head_idx].masked_fill_(causal_mask, float("-inf"))

            elif mask_type < 0:
                # Extract padding masks for current batch (if they exist)
                query_mask_batch = None
                key_mask_batch = None
                if query_padding_mask is not None:
                    query_mask_batch = query_padding_mask[batch_idx, :seqlen_q]
                if key_padding_mask is not None:
                    key_mask_batch = key_padding_mask[batch_idx, :seqlen_k]

                streaming_mask = construct_streaming_mask(
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    sink_size=sink_size,
                    local_size=local_size,
                    is_causal=is_causal,
                    query_padding_mask=query_mask_batch,
                    key_padding_mask=key_mask_batch,
                    device=device,
                )

                scores[head_idx].masked_fill_(streaming_mask, float("-inf"))

            elif mask_type > 0:
                # This indicates Block Sparse Attention, not considered for now
                pass

        attn = torch.softmax(scores, dim=-1).to(v_batch.dtype)

        if query_padding_mask is not None:
            query_mask = query_padding_mask[batch_idx, :seqlen_q].to(device)
            attn = attn.masked_fill(~query_mask[None, :, None], 0.0)

        # Apply dropout
        if p_dropout > 0.0:
            if dropout_mask is not None:
                # Use predefined dropout mask (for testing)
                attn_drop = attn.masked_fill(
                    ~dropout_mask[batch_idx, :, :seqlen_q, :seqlen_k], 0.0
                )
            else:
                # Random dropout
                drop_mask = torch.rand_like(attn) > p_dropout
                attn_drop = attn.masked_fill(~drop_mask, 0.0)
            dropout_scale = 1.0 / (1.0 - p_dropout)
        else:
            attn_drop = attn
            dropout_scale = 1.0

        out_batch = torch.einsum("hqk,khd->qhd", attn_drop, v_batch * dropout_scale)

        if query_padding_mask is not None:
            query_mask = query_padding_mask[batch_idx, :seqlen_q].to(device)
            out_batch = out_batch.masked_fill(~query_mask[:, None, None], 0.0)

        output[q_start:q_end] = out_batch

        if return_attn_probs:
            attn_weight_lists.append(attn)

    if return_attn_probs:
        return output, attn_weight_lists
    else:
        return output, None
