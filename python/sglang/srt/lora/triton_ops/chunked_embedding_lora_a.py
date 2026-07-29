import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _chunked_embedding_lora_a_kernel(
    # Pointers to tensors
    input_ids,
    weights,
    output,
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
    # Chunk info
    seg_indptr,
    weight_indices,
    lora_ranks,
    num_segments,
    permutation,
    # Meta-parameters
    BLOCK_RANK: tl.constexpr,
):
    """
    Embedding lookup for LoRA A weights without support for extra tokens.

    Each program handles one chunk of tokens across rank dimension
    """
    chunk_idx = tl.program_id(axis=0)
    # If chunk id is larger than actual number of chunks, skip
    if chunk_idx >= num_segments:
        return
    # Load LoRA adapter index for this segment, then look up the rank
    lora_index = tl.load(weight_indices + chunk_idx)
    rank_val = tl.load(lora_ranks + lora_index)
    # If rank is 0, skip
    if rank_val == 0:
        return
    # for each token in chunk, load embedding across rank dimension
    chunk_start = tl.load(seg_indptr + chunk_idx)
    chunk_end = tl.load(seg_indptr + chunk_idx + 1)
    for c in range(chunk_start, chunk_end):
        s_index = tl.load(permutation + c)
        # Load the token ID
        token_id = tl.load(input_ids + s_index)
        # Process in chunks of BLOCK_RANK dimensions
        num_blocks = tl.cdiv(rank_val, BLOCK_RANK)

        for block_id in range(num_blocks):
            rank_offset = tl.arange(0, BLOCK_RANK) + block_id * BLOCK_RANK
            rank_mask = rank_offset < rank_val

            # Use regular LoRA A weights
            # weights shape: (num_loras, rank, vocab_size)
            # We need to load weights[lora_index, rank_offset, token_id]
            weight_ptr = (
                weights
                + lora_index * w_stride_0
                + rank_offset * w_stride_1
                + token_id * w_stride_2
            )
            emb_values = tl.load(weight_ptr, mask=rank_mask, other=0.0)

            # Write to output
            output_ptr = (
                output + s_index * output_stride_0 + rank_offset * output_stride_1
            )
            tl.store(output_ptr, emb_values, mask=rank_mask)


def chunked_embedding_lora_a_forward(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
) -> torch.Tensor:
    """
    Chunked Forward pass for LoRA A embedding lookup; each program handles one chunk of embedding lookup work
    belonging to the same adapter

    Args:
        input_ids: (s,) token IDs
        weights: (num_loras, rank, vocab_size) LoRA A embedding weights
        batch_info: LoRABatchInfo containing batch information
        vocab_size: base vocabulary size

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
    num_segments = batch_info.num_segments
    # 1D Grid: one program per chunk of embedding lookup work
    grid = (batch_info.bs if batch_info.use_cuda_graph else num_segments,)
    output = torch.zeros((S, rank), device=input_ids.device, dtype=weights.dtype)

    _chunked_embedding_lora_a_kernel[grid](
        input_ids,
        weights,
        output,
        vocab_size,
        rank,
        num_loras,
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.num_segments,
        batch_info.permutation,
        BLOCK_RANK,
    )

    return output
