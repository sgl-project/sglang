import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions
from sglang.srt.lora.utils import LoRABatchInfo


@triton.jit
def _sgemm_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    S,  # total number of rows in x/output (used for OOB clamping)
    N,  # output_dim
    K,  # r
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
    sorted_token_ids,
    # Meta parameters
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # For fused output scaling
    scalings,
):
    """
    Computes a segmented batched matrix multiplication for the LoRA B matrix
    and adds the result to the output in-place.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (torch.Tensor): The intermediate tensor from the LoRA 'A' multiplication,
            of shape `(s, K)`, where `s` is the total number of tokens.
        weights (torch.Tensor): The LoRA 'B' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`. This can be
            the base model's output for a fused add operation.
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    # See sgemm_lora_a for why we clamp masked-lane indices.
    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    safe_n = tl.minimum(n_offset, N - 1)

    # Iterate to compute the block in output matrix
    n_mask = n_offset[None, :] < N
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k * BLOCK_K + k_offset
        k_mask = cur_k < K
        safe_k = tl.minimum(cur_k, K - 1)
        x_tile = tl.load(
            x + safe_row[:, None] * x_stride_0 + safe_k[None, :] * x_stride_1,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            weights
            + w_index * w_stride_0
            + safe_k[:, None] * w_stride_2
            + safe_n[None, :] * w_stride_1,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (
        output + safe_row[:, None] * output_stride_0 + safe_n[None, :] * output_stride_1
    )
    output_mask = row_mask[:, None] & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_b_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor = None,
) -> torch.Tensor:
    # x: (s, max_r)
    # weights: (num_lora, output_dim, max_r)
    # output: (s, output_dim)
    # output_dim is much larger than max_r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    N = weights.shape[-2]
    R = weights.shape[-1]
    assert x.shape[-1] == R

    # Block shapes
    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_N = 256

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
        S,
        N,
        R,
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
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_N,
        BLOCK_R,
        batch_info.scalings,
    )
    return output
