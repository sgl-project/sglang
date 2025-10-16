from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel


@cached_triton_kernel(lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"]))
@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_expand_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # For fused output scaling
    scalings,
    # Offsets of q/k/v slice on output dimension
    slice_offsets,
    # Meta parameters
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,  # K = R
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA expand operations.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, num_slices * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, output_dim, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, output_dim).
    """
    tl.static_assert(NUM_SLICES <= 3)

    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1

    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = OUTPUT_DIM
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    slice_id = tl.program_id(axis=1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)

    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    cur_rank = tl.minimum(MAX_RANK, cur_rank)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_expand_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor],
) -> torch.Tensor:

    # x: (s, slice_num * r)
    # weights: (num_lora, output_dim, r)
    # slice_offsets: boundaries for different slices in the output dimension
    # output: (s, output_dim)

    # Compute lora_output with shape (s, output_dim) as follows:
    # For each slice i, accumulates:
    # lora_output[:, slice_offsets[i]:slice_offsets[i+1]] += scaling * sgemm(x[:, i*cur_rank:(i+1)*cur_rank], weights[:, slice_offsets[i]:slice_offsets[i+1], :])

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Get dims
    M = x.shape[0]
    input_dim = x.shape[1]
    OUTPUT_DIM = weights.shape[1]
    MAX_RANK = weights.shape[2]
    num_slices = len(slice_offsets) - 1
    assert input_dim == num_slices * MAX_RANK

    # TODO (lifuhuang): fine-tune per operation
    BLOCK_M = batch_info.max_len
    BLOCK_K = 16
    BLOCK_N = 64

    num_segments = batch_info.num_segments

    grid = (
        triton.cdiv(max_slice_size, BLOCK_N),
        num_slices,  # number of slices in the input/output
        batch_info.bs if batch_info.use_cuda_graph else num_segments,
    )

    if base_output is None:
        output = torch.zeros((M, OUTPUT_DIM), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _chunked_lora_expand_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=num_segments,
        scalings=batch_info.scalings,
        slice_offsets=slice_offsets,
        # constants
        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUTPUT_DIM,
        MAX_RANK=MAX_RANK,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
