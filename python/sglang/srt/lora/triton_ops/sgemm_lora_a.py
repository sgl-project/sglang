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
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    # x: (s, K), s is the sum of sequence lengths
    # weights: (num_lora, N, K)
    # output: (s, N)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    rank = tl.load(lora_ranks + w_index)
    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    N = tl.minimum(N, rank * stack_num)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = (x + seg_start * x_stride_0) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iteate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len)
            and (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) and (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) and (n_offset[None, :] < N)
    tl.store(output_ptr, partial_sum, mask=output_mask)


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

    S = x.shape[0]
    R = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    # Block shapes
    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = 16

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
        batch_info.bs,
    )

    output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        R,
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
        BLOCK_S,
        BLOCK_R,
        BLOCK_K,
    )
    return output
