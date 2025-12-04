#############################
#########cuda graph##########
#############################
import torch
import triton
import triton.language as tl
from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _embedding_extra_tokens_kernel(
    # Pointers to tensors
    input_ids,
    output,
    extra_embeddings,
    # Dimensions
    vocab_size,
    embed_dim,
    num_loras,
    # Strides
    output_stride_0,
    output_stride_1,
    extra_emb_stride_0,  # stride for lora index
    extra_emb_stride_1,  # stride for token
    extra_emb_stride_2,  # stride for embed dim
    # Batch info
    seg_lens,
    seg_indptr,
    weight_indices,
    # Meta-parameters
    BLOCK_EMBED: tl.constexpr,
):
    """
    Embedding lookup for extra/added tokens (tokens >= vocab_size).
    Each program handles one token across a block of embedding dimensions.
    Grid: (max_len, bs)
    """
    batch_id = tl.program_id(axis=1)
    token_idx = tl.program_id(axis=0)
    
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    
    # Check if this token is within the segment
    if token_idx >= seg_len:
        return
    
    # Load the token ID
    token_id = tl.load(input_ids + seg_start + token_idx)
    
    # Check if this is an extra token
    is_extra_token = token_id >= vocab_size
    
    if not is_extra_token:
        return  # Skip non-extra tokens
    
    # Calculate extra token ID
    extra_token_id = token_id - vocab_size
    
    # Process in chunks of BLOCK_EMBED dimensions
    num_blocks = tl.cdiv(embed_dim, BLOCK_EMBED)
    
    for block_id in range(num_blocks):
        embed_offset = tl.arange(0, BLOCK_EMBED) + block_id * BLOCK_EMBED
        embed_mask = embed_offset < embed_dim
        
        # Load from extra embeddings
        # extra_embeddings shape: (num_loras, num_extra_tokens, embed_dim)
        extra_emb_ptr = (
            extra_embeddings
            + w_index * extra_emb_stride_0
            + extra_token_id * extra_emb_stride_1
            + embed_offset * extra_emb_stride_2
        )
        emb_values = tl.load(extra_emb_ptr, mask=embed_mask, other=0.0)
        
        # Write to output (overwrite the position)
        output_ptr = (
            output
            + (seg_start + token_idx) * output_stride_0
            + embed_offset * output_stride_1
        )
        tl.store(output_ptr, emb_values, mask=embed_mask)


def embedding_extra_tokens_fwd(
    input_ids: torch.Tensor,
    output: torch.Tensor,  # Will be modified in-place
    extra_embeddings: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
) -> torch.Tensor:
    """
    Forward pass for extra token embedding lookup (in-place operation).
    
    Args:
        input_ids: (s,) token IDs
        output: (s, embed_dim) output tensor to be modified in-place
        extra_embeddings: (num_loras, num_extra_tokens, embed_dim) extra token embeddings
        batch_info: LoRABatchInfo containing batch information
        vocab_size: base vocabulary size
        
    Returns:
        output: (s, embed_dim) modified output tensor
    """
    assert input_ids.is_contiguous()
    assert output.is_contiguous()
    assert extra_embeddings.is_contiguous()
    assert len(input_ids.shape) == 1
    assert len(output.shape) == 2
    assert len(extra_embeddings.shape) == 3
    
    S = input_ids.shape[0]
    embed_dim = output.shape[1]
    num_loras = extra_embeddings.shape[0]
    
    # Block size for embedding dimension
    BLOCK_EMBED = 128
    
    extra_emb_stride = (
        extra_embeddings.stride(0),
        extra_embeddings.stride(1),
        extra_embeddings.stride(2),
    )
    
    # Grid: one program per token in each batch segment
    grid = (
        batch_info.max_len,
        batch_info.bs,
    )
    
    _embedding_extra_tokens_kernel[grid](
        input_ids,
        output,
        extra_embeddings,
        vocab_size,
        embed_dim,
        num_loras,
        output.stride(0),
        output.stride(1),
        extra_emb_stride[0],
        extra_emb_stride[1],
        extra_emb_stride[2],
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        BLOCK_EMBED,
    )
    
    return output
#############################
#############################
#############################