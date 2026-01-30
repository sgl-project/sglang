import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _qkv_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Parameters of size
    K,  # K = R
    max_qkv_out_dim,  # max(output_q_dim, output_kv_dim)
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths and weight id
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    # Offsets of q/k/v slice on output dimension
    n_offs,
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # For fused output scaling
    scalings,
):
    """
    This kernel packs 3 sgemms (q/k/v) into a single kernel. The multiplication
    results are accumulated into the output tensor.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, 3 * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank. The second dimension is partitioned
            for Q, K, and V.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, N_Q + 2 * N_KV, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, N_Q + 2 * N_KV).
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (x + seg_start * x_stride_0 + (qkv_id * K) * x_stride_1) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < n_size),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (output + seg_start * output_stride_0 + n_start * output_stride_1) + (
        s_offset[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def qkv_lora_b_fwd(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor = None,
) -> torch.Tensor:

    # x: (s, 3 * r)
    # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
    # output_offset = [0, output_dim_q, output_dim_q + output_dim_kv,
    #                     output_dim_q + 2 * output_dim_kv]
    # max_qkv_out_dim = max(output_dim_q, output_dim_kv)
    # output: (s, output_dim_q + 2 * output_dim_kv)

    # Compute lora_output with shape (s, output_dim) as follows:
    # lora_output[:, :output_dim_q] = sgemm(x[:, :r], qkv_lora_b[:, :outptu_dim_q, :])
    # lora_output[:, output_dim_q: output_dim_q + output_dim_kv]
    #      = sgemm(x[:, r: 2 * r], qkv_lora_b[:, outptu_dim_q: output_dim_q + output_dim_kv, :])
    # lora_output[:, output_dim_q + output_dim_kv: ]
    #      = sgemm(x[:, 2 * r: , qkv_lora_b[:, output_dim_q + output_dim_kv: , :])

    # Get dims
    s = x.shape[0]
    input_dim = x.shape[1]
    r = qkv_lora_b.shape[-1]
    output_dim = qkv_lora_b.shape[-2]
    assert input_dim == 3 * r
    assert output_offset.shape[0] == 4

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S)
        * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
        3,  # this dimension decides current block computes on q, k or v
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _qkv_lora_b_kernel[grid_b](
        x,
        qkv_lora_b,
        output,
        r,
        max_qkv_out_dim,
        x.stride(0),
        x.stride(1),
        qkv_lora_b.stride(0),
        qkv_lora_b.stride(1),
        qkv_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        output_offset,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
    )

    return output
