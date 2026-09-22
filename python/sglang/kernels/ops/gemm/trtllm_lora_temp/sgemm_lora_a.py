import functools

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
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
    SPLIT_K: tl.constexpr = 1,
    ENABLE_PDL: tl.constexpr = False,
    PADDED_RANK: tl.constexpr = False,
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
    # Fold the split-K factor out of axis-0 (SPLIT_K == 1 -> pid_sk == 0, pid_tile == pid).
    pid_sk = pid % SPLIT_K
    pid_tile = pid // SPLIT_K

    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    # Adjust N (stack_num * max_rank) to this adapter's actual rank.
    if not PADDED_RANK:
        N = tl.minimum(N, rank * stack_num)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid_tile // num_pid_n
    pid_n = pid_tile % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate.
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # GDC wait: ensure the prior kernel (producer of x) has fully completed
    # before consuming its output.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * SPLIT_K * x_stride_1
        w_ptrs += BLOCK_K * SPLIT_K * w_stride_2

    # All input reads are done; hint the runtime to launch the dependent kernel.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Store result to output matrix
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    if SPLIT_K == 1:
        tl.store(output_ptr, partial_sum.to(output.dtype.element_ty), mask=output_mask)
    else:
        tl.atomic_add(
            output_ptr,
            partial_sum.to(output.dtype.element_ty),
            mask=output_mask,
            sem="relaxed",
        )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
    out_alloc_stream=None,
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

    if (
        lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
        or lora_envs.SGLANG_OPT_LORA_CUBLAS_A.get()
    ) and weights.shape[
        0
    ] == 1:  # single-adapter fast path: only valid with one resident slot
        # Honor out_alloc_stream like the Triton path below: under SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC
        # the shrink output must be allocated on the MAIN (consumer) stream so the caching allocator
        # frees/reuses it on the consumer's schedule (cuda-graph WAR). F.linear has no out=, so
        # allocate explicitly and matmul into it.
        if out_alloc_stream is not None:
            with torch.cuda.stream(out_alloc_stream):
                output = torch.empty((S, R), device=x.device, dtype=x.dtype)
        else:
            output = torch.empty((S, R), device=x.device, dtype=x.dtype)
        return torch.matmul(x, weights[0].transpose(-2, -1), out=output)

    # Block shapes
    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = triton.next_power_of_2(R)

    sorted_by_adapter = batch_info.permutation is not None

    num_s_tiles = triton.cdiv(batch_info.max_len, BLOCK_S)
    split_k = 1
    if lora_envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.get() and x.is_cuda:
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        base_grid = batch_info.bs * num_s_tiles
        num_sms = _num_sms(x.device.index)
        if base_grid < num_sms and num_k_tiles >= 16:
            split_k = max(1, min(2 * num_sms // base_grid, num_k_tiles, 16))

    launch_kwargs = {}
    if split_k > 1:
        if out_alloc_stream is not None:
            with torch.cuda.stream(out_alloc_stream):
                output = torch.empty((S, R), device=x.device, dtype=torch.float32)
            output.zero_()
        else:
            output = torch.zeros((S, R), device=x.device, dtype=torch.float32)
        launch_kwargs = {
            "num_warps": 2 if split_k <= 4 else 4,
            "num_stages": 3,
        }
    elif out_alloc_stream is not None:
        # Allocate the output on the MAIN (consumer) stream when requested, so the caching allocator
        # frees/reuses it on the consumer's schedule, not the side stream's (cuda-graph WAR — see
        # lora_overlap_alloc_stream / SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC).
        with torch.cuda.stream(out_alloc_stream):
            output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    else:
        output = torch.empty((S, R), device=x.device, dtype=x.dtype)

    grid = (
        num_s_tiles * split_k,
        batch_info.bs,
    )

    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
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
        split_k,
        ENABLE_PDL=enable_pdl,
        **launch_kwargs,
        **pdl_kwargs,
    )
    # split_k>1 returns the fp32 accumulator directly; the LoRA-B expand casts x to the weight dtype
    # on-load (fused), dropping the standalone fp32->bf16 copy kernel. split_k==1 already returns x.dtype.
    return output


def shared_sink_sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    *,
    stack_num: int,
    padded_rank: bool,
    out_alloc_stream=None,
) -> torch.Tensor:
    """Shared-sink shrink with the measured fixed-width schedule."""
    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert x.ndim == 2
    assert weights.ndim == 3

    num_tokens = x.shape[0]
    rank_width = weights.shape[-2]
    input_width = weights.shape[-1]
    assert x.shape[-1] == input_width

    block_s = 16
    block_k = 256
    block_rank = 16
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * triton.cdiv(rank_width, block_rank),
        batch_info.bs,
    )

    if out_alloc_stream is None:
        output = torch.empty((num_tokens, rank_width), device=x.device, dtype=x.dtype)
    else:
        with torch.cuda.stream(out_alloc_stream):
            output = torch.empty(
                (num_tokens, rank_width), device=x.device, dtype=x.dtype
            )

    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        rank_width,
        input_width,
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
        batch_info.permutation is not None,
        block_s,
        block_rank,
        block_k,
        PADDED_RANK=padded_rank,
    )
    return output
