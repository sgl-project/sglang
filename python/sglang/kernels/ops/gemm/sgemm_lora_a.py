import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.kernel_utils import _resolve_token_positions
from sglang.srt.environ import envs
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
    sorted_token_ids,
    # Meta parameters
    SORTED_BY_ADAPTER: tl.constexpr,
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

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    N = tl.minimum(N, rank * stack_num)

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
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
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
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
    output: torch.Tensor = None,
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
    if splitk_applies(weights, batch_info):
        return _sgemm_lora_a_fwd_splitk(
            x, weights, batch_info, stack_num, output=output
        )

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

    sorted_by_adapter = batch_info.permutation is not None

    if output is None:
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
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_R,
        BLOCK_K,
    )
    return output


# Decode split-K variant: the stock grid gives each of bs single-token segments
# one program streaming the whole r x K adapter. fp32 partials over K blocks are
# summed in a fixed order by a second kernel (deterministic, no atomics).
_SPLITK_ENABLED = envs.SGLANG_OPT_LORA_DECODE_FUSED_LAUNCHES.get()
_SPLITK_MIN_K = 2048
_SPLITK_BLOCK_K = 256


@triton.jit
def _sgemm_lora_a_splitk_partial_kernel(
    x,
    weights,
    partial,
    N,
    K,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    p_stride_k,
    p_stride_s,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    stack_num,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    N = tl.minimum(N, rank * stack_num)
    if pid_n * BLOCK_N >= N:
        return
    s_offset = tl.arange(0, BLOCK_S)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K) + pid_k * BLOCK_K
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_tile = tl.load(
        x + s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(
        weights
        + w_index * w_stride_0
        + k_offset[:, None] * w_stride_2
        + n_offset[None, :] * w_stride_1,
        mask=(k_offset[:, None] < K) & (n_offset[None, :] < N),
        other=0.0,
    )
    acc = tl.dot(x_tile, w_tile)
    p_ptr = (
        partial
        + pid_k * p_stride_k
        + s_physical[:, None] * p_stride_s
        + n_offset[None, :]
    )
    tl.store(p_ptr, acc, mask=(s_offset[:, None] < seg_len) & (n_offset[None, :] < N))


@triton.jit
def _sgemm_lora_a_splitk_reduce_kernel(
    partial,
    output,
    N,
    KSPLIT,
    p_stride_k,
    p_stride_s,
    o_stride_0,
    o_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    stack_num,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    N = tl.minimum(N, rank * stack_num)
    if pid_n * BLOCK_N >= N:
        return
    s_offset = tl.arange(0, BLOCK_S)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    acc = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for kk in range(0, KSPLIT):
        acc += tl.load(
            partial
            + kk * p_stride_k
            + s_physical[:, None] * p_stride_s
            + n_offset[None, :],
            mask=mask,
            other=0.0,
        )
    tl.store(
        output + s_physical[:, None] * o_stride_0 + n_offset[None, :] * o_stride_1,
        acc.to(output.dtype.element_ty),
        mask=mask,
    )


def splitk_applies(weights: torch.Tensor, batch_info: LoRABatchInfo) -> bool:
    """Whether sgemm_lora_a_fwd takes the decode split-K variant."""
    return (
        _SPLITK_ENABLED
        and batch_info.max_len <= 16
        and weights.shape[-1] >= _SPLITK_MIN_K
    )


def splitk_partial_shape(x: torch.Tensor, weights: torch.Tensor):
    """Shape of the fp32 partials buffer the split-K variant fills."""
    return (triton.cdiv(x.shape[1], _SPLITK_BLOCK_K), x.shape[0], weights.shape[-2])


def _sgemm_lora_a_fwd_splitk(
    x, weights, batch_info, stack_num, output=None, partial=None
):
    S, K = x.shape
    R = weights.shape[-2]
    BLOCK_S, BLOCK_R, BLOCK_K = 16, 16, _SPLITK_BLOCK_K
    ksplit = triton.cdiv(K, BLOCK_K)
    sorted_by_adapter = batch_info.permutation is not None
    if output is None:
        output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    if partial is None:
        partial = torch.empty((ksplit, S, R), device=x.device, dtype=torch.float32)
    meta = (
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
    )
    _sgemm_lora_a_splitk_partial_kernel[
        (triton.cdiv(R, BLOCK_R), batch_info.bs, ksplit)
    ](
        x,
        weights,
        partial,
        R,
        K,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        partial.stride(0),
        partial.stride(1),
        *meta,
        stack_num,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_R,
        BLOCK_K=BLOCK_K,
    )
    _sgemm_lora_a_splitk_reduce_kernel[(triton.cdiv(R, BLOCK_R), batch_info.bs)](
        partial,
        output,
        R,
        ksplit,
        partial.stride(0),
        partial.stride(1),
        output.stride(0),
        output.stride(1),
        *meta,
        stack_num,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_R,
    )
    return output
