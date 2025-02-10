import torch
import triton
import triton.language as tl

from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _gate_up_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Parameters of size
    K,  # K = R
    output_dim,
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
    # Meta parameters
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # For fused output scaling and adding
    fuse_scaling_add,
    scaling,
):
    # This kernel packs 2 sgemms (gate/up) into a single kernel.

    # x: (s, 2 * K), s is the sum of sequence lengths, K equals to lora rank
    # weights: (num_lora, 2 * output_dim, K)
    # output: (s, 2 * output_dim)
    # output_dim >> K

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # gate_up_id decides which of gate or up (0: gate, 1: up)
    batch_id = tl.program_id(axis=2)
    gate_up_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    w_index = tl.load(weight_indices + batch_id)
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = gate_up_id * output_dim  # offset on output dim

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(output_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first block of x and weights
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (x + seg_start * x_stride_0 + (gate_up_id * K) * x_stride_1) + (
        s_offset[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
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
            mask=(k_offset[:, None] < K - k * BLOCK_K)
            and (n_offset[None, :] < output_dim),
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
    output_mask = (s_offset[:, None] < seg_len) and (n_offset[None, :] < output_dim)
    if fuse_scaling_add:
        partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def gate_up_lora_b_fwd(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor = None,
    scaling: float = 1.0,
) -> torch.Tensor:

    # x: (s, 2 * r)
    # gate_up_lora_b: (num_lora, 2 * output_dim, r)
    # output: (s, 2 * output_dim)

    # Compute lora_output with shape (s, output_dim) as follows:
    # lora_output[:, :output_dim] = sgemm(x[:, :r], gate_up_lora_b[:, :output_dim, :])
    # lora_output[:, output_dim:]
    #      = sgemm(x[:, r:], gate_up_lora_b[:, output_dim:, :])

    # Get dims
    s = x.shape[0]
    input_dim = x.shape[1]
    r = gate_up_lora_b.shape[-1]
    assert input_dim == 2 * r

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(output_dim, BLOCK_OUT),
        2,  # this dimension decides current block computes on gate or up proj
        batch_info.bs,
    )

    if base_output is None:
        output = torch.empty((s, 2 * output_dim), device=x.device, dtype=x.dtype)
        fuse_scaling_add = False
    else:
        output = base_output
        fuse_scaling_add = True

    _gate_up_lora_b_kernel[grid_b](
        x,
        gate_up_lora_b,
        output,
        r,
        output_dim,
        x.stride(0),
        x.stride(1),
        gate_up_lora_b.stride(0),
        gate_up_lora_b.stride(1),
        gate_up_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        fuse_scaling_add,
        scaling,
    )

    return output
