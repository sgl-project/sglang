import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _sgemm_lora_a_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    N,  # stack_num * r
    K,  # input_dim
    stack_num,
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths,ranks and weight id
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    ordered_indices,
    chunk_to_weight,
    cu_chunk_lens,
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a segmented batched matrix multiplication for the LoRA A matrix.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * stack_num]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA 'A' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    chunk_id = tl.program_id(1)
    slice_id = tl.program_id(0)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(chunk_to_weight + chunk_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    seg_start = tl.load(cu_chunk_lens + chunk_id)
    seg_end = tl.load(cu_chunk_lens + chunk_id + 1)

    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    N = tl.minimum(N, rank * stack_num)

    # The tile in output matrix will have (pid_s, pid_n) as id

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset_orig = tl.arange(0, BLOCK_S) + seg_start
    s_offset = tl.load(ordered_indices + s_offset_orig, mask=s_offset_orig < seg_end, other=0)  # (BLOCK_S,)

    n_offset = tl.arange(0, BLOCK_N) + slice_id * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_end) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output  + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_end) & (n_offset[None, :] < N)
    tl.store(output_ptr, partial_sum, mask=output_mask)

def reorder_and_prepare_chunks(weight_indices, seg_lens, chunk_size: int, device: torch.device = torch.device("cpu")):
    # Create a weight index for each row by repeating weight_indices according to seg_lens
    row_weight_indices = torch.repeat_interleave(weight_indices, seg_lens)
    
    # Sort rows by weight index (stable sort keeps relative order within each weight)
    reorder_indices = torch.argsort(row_weight_indices, stable=True)
    
    # Get reordered weights to find group boundaries
    weights_reordered = row_weight_indices[reorder_indices]
    
    # Get unique weights and their counts
    unique_weights, counts = torch.unique_consecutive(weights_reordered, return_counts=True)
    
    # Build chunk arrays
    chunk_to_weight = []
    cu_chunk_lens = [0]
    
    cumulative_pos = 0
    for weight_idx, group_len in zip(unique_weights, counts):
        group_len = group_len.item()
        num_chunks = (group_len + chunk_size - 1) // chunk_size
        
        chunk_to_weight.extend([weight_idx.item()] * num_chunks)
        
        # Add boundaries for each chunk
        for i in range(1, num_chunks):
            cu_chunk_lens.append(cumulative_pos + i * chunk_size)
        cu_chunk_lens.append(cumulative_pos + group_len)
        
        cumulative_pos += group_len
    
    chunk_to_weight = torch.tensor(chunk_to_weight, dtype=torch.int32, device=device)
    cu_chunk_lens = torch.tensor(cu_chunk_lens, dtype=torch.int32, device=device)
    
    return reorder_indices, chunk_to_weight, cu_chunk_lens


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, stack_num * r, input_dim)
    # output: (s, stack_num * r)
    # stack_num: run_qkv_lora: 3, run_gate_up_lora: 2
    # when called by run_qkv_lora, the weights.shape[-2] will be 3 * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Block shapes
    BLOCK_S = 16
    BLOCK_N = 16
    BLOCK_K = 256

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K

    grid = (
        triton.cdiv(N, BLOCK_N),
        len(batch_info.chunk_to_weight),
    )

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        N,
        K,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.ordered_indices,
        batch_info.chunk_to_weight,
        batch_info.cu_chunk_lens,
        BLOCK_S,
        BLOCK_N,
        BLOCK_K,
    )

    return output
