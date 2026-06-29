import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _embedding_lora_a_kernel(
    # Pointers to tensors
    input_ids,
    weights,
    output,
    extra_embeddings,
    # Dimensions
    vocab_size,
    rank,
    num_loras,
    # Strides
    w_stride_0,  # stride for lora index
    w_stride_1,  # stride for rank
    w_stride_2,  # stride for vocab
    output_stride_0,
    output_stride_1,
    extra_emb_stride_0,  # stride for lora index
    extra_emb_stride_1,  # stride for token
    extra_emb_stride_2,  # stride for hidden dim (= rank for extra embeddings)
    # Batch info
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    # Meta-parameters
    BLOCK_RANK: tl.constexpr,
    HAS_EXTRA_EMBEDDINGS: tl.constexpr,
):
    """
    Embedding lookup for LoRA A weights with support for extra tokens.

    Each program handles one block of rank dimensions for one token.

    Grid: (max_len, bs, cdiv(rank, BLOCK_RANK))
    - axis 0: token index within the segment
    - axis 1: batch (segment) index
    - axis 2: rank block index (parallelizes the rank dimension)
    """
    token_idx = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    rank_block_id = tl.program_id(axis=2)

    w_index = tl.load(weight_indices + batch_id)
    rank_val = tl.load(lora_ranks + w_index)

    # Early exit: no work for this rank block
    if rank_block_id * BLOCK_RANK >= rank_val:
        return

    # Compute rank offsets for this block
    rank_offset = tl.arange(0, BLOCK_RANK) + rank_block_id * BLOCK_RANK
    rank_mask = rank_offset < rank_val

    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)

    # Check if this token is within the segment
    if token_idx >= seg_len:
        return

    # Load the token ID
    token_id = tl.load(input_ids + seg_start + token_idx)

    # Check if this is an extra token
    is_extra_token = token_id >= vocab_size

    if HAS_EXTRA_EMBEDDINGS and is_extra_token:
        # Use extra embeddings
        extra_token_id = token_id - vocab_size
        extra_emb_ptr = (
            extra_embeddings
            + w_index * extra_emb_stride_0
            + extra_token_id * extra_emb_stride_1
            + rank_offset * extra_emb_stride_2
        )
        emb_values = tl.load(extra_emb_ptr, mask=rank_mask, other=0.0)
    else:
        # Use regular LoRA A weights
        # weights shape: (num_loras, rank, vocab_size)
        # We need to load weights[w_index, rank_offset, token_id]
        token_id_clamped = tl.minimum(token_id, vocab_size - 1)
        weight_ptr = (
            weights
            + w_index * w_stride_0
            + rank_offset * w_stride_1
            + token_id_clamped * w_stride_2
        )
        emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)

    # Write to output
    output_ptr = (
        output
        + (seg_start + token_idx) * output_stride_0
        + rank_offset * output_stride_1
    )
    tl.store(output_ptr, emb_values, mask=rank_mask)


def embedding_lora_a_fwd(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
    extra_embeddings: torch.Tensor = None,
) -> torch.Tensor:
    """
    Forward pass for LoRA A embedding lookup.

    Args:
        input_ids: (s,) token IDs
        weights: (num_loras, rank, vocab_size) LoRA A embedding weights
        batch_info: LoRABatchInfo containing batch information
        vocab_size: base vocabulary size
        extra_embeddings: (num_loras, num_extra_tokens, rank) extra token embeddings

    Returns:
        output: (s, rank) embedded features
    """
    assert input_ids.is_contiguous()
    assert weights.is_contiguous()
    assert len(input_ids.shape) == 1
    assert len(weights.shape) == 3

    S = input_ids.shape[0]
    num_loras = weights.shape[0]
    rank = weights.shape[1]

    # Block size for rank dimension
    BLOCK_RANK = 128

    has_extra_embeddings = extra_embeddings is not None
    if has_extra_embeddings:
        assert extra_embeddings.is_contiguous()
        extra_emb_stride = (
            extra_embeddings.stride(0),
            extra_embeddings.stride(1),
            extra_embeddings.stride(2),
        )
    else:
        # Create dummy tensor to satisfy Triton
        extra_embeddings = torch.empty(
            (1, 1, 1), device=input_ids.device, dtype=weights.dtype
        )
        extra_emb_stride = (1, 1, 1)

    # Grid: parallelize across tokens, batches, AND rank blocks
    num_rank_blocks = triton.cdiv(rank, BLOCK_RANK)
    grid = (
        batch_info.max_len,
        batch_info.bs,
        num_rank_blocks,
    )

    output = torch.zeros(
        (S, rank), device=input_ids.device, dtype=weights.dtype
    )

    _embedding_lora_a_kernel[grid](
        input_ids,
        weights,
        output,
        extra_embeddings,
        vocab_size,
        rank,
        num_loras,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        extra_emb_stride[0],
        extra_emb_stride[1],
        extra_emb_stride[2],
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        BLOCK_RANK,
        has_extra_embeddings,
        num_warps=4,
    )

    return output