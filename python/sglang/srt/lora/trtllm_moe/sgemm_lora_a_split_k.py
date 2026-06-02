"""Opt-in fp32 split-K for the dense LoRA-A (shrink) GEMM.

Vendored from sgl-project/sglang PR #26962 (https://github.com/sgl-project/sglang/pull/26962) so the
split-K path lives in one file behind the ``SGLANG_ENABLE_LORA_SHRINK_SPLIT_K`` gate; the base
``triton_ops/sgemm_lora_a.py`` routes here only when that env is set (default off → original kernel).

At decode the shrink grid is ~bs programs (one N-block/seq; rank is tiny), far below the SM count, so the
large-K reduction under-fills the GPU. This splits K across SPLIT_K programs that fp32-atomic-add into a
pre-zeroed accumulator, then casts back to the activation dtype. SPLIT_K==1 (prefill / large batch / small
K, or env off) is the original single-program path, bit-identical to the base kernel.
"""
import functools

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions
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
):
    """
    Computes a segmented batched matrix multiplication for the LoRA A matrix.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * stack_num]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    SPLIT_K (constexpr, default 1) splits the K reduction across SPLIT_K programs
    for the under-filled decode shape (the shrink grid is ~bs programs, far below
    the SM count, so a large-K reduction leaves SMs idle). With SPLIT_K > 1 each
    program walks a strided K-slice and atomic-adds its partial sum into `output`,
    which MUST then be fp32 (and pre-zeroed): bf16 atomic_add is an emulated CAS
    loop that contends and loses precision, whereas native fp32 atomics scale and
    are exact -- the launcher casts the fp32 result back to the activation dtype.
    SPLIT_K == 1 is the plain single-program store path (compile-time identical to
    the non-split kernel; constexpr folds the split-only expressions away).

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

    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    N = tl.minimum(N, rank * stack_num)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid_tile // num_pid_n
    pid_n = pid_tile % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate. With SPLIT_K > 1 this program starts at K-tile pid_sk and
    # strides by BLOCK_K * SPLIT_K; with SPLIT_K == 1 it walks all of K.
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

    # Store result to output matrix
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    if SPLIT_K == 1:
        tl.store(output_ptr, partial_sum.to(output.dtype.element_ty), mask=output_mask)
    else:
        # fp32 accumulator, pre-zeroed; combine the K-splits with native fp32 atomics.
        tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def sgemm_lora_a_fwd_split_k(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
    out_alloc_stream=None,
) -> torch.Tensor:
    # out_alloc_stream: accepted for signature parity with sgemm_lora_a_fwd. Main-stream output
    # allocation (SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC) is not wired into the split-K path — no current
    # config exercises split-K together with the two-stream overlap on a mamba model (qwen3.5 leaves
    # split-K off; kimi uses split-K but is coherent with a single stream — no mamba). Wire it here if
    # that combination ever ships.
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

    sorted_by_adapter = batch_info.permutation is not None

    # Opt-in split-K for the under-filled decode shape: the shrink grid is ~bs
    # programs, far below the SM count, so a large-K reduction leaves SMs idle.
    # Target ~2x SM oversubscription, capped by the K-tile count and at 16 (the
    # win plateaus by sk16-24). split_k stays 1 (the plain store path, compile-time
    # identical to before) for prefill, large batch, or small K, and whenever the
    # flag is off. See SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.
    split_k = 1
    if envs.SGLANG_ENABLE_LORA_SHRINK_SPLIT_K.get() and x.is_cuda:
        num_k_tiles = triton.cdiv(K, BLOCK_K)
        base_grid = batch_info.bs * triton.cdiv(batch_info.max_len, BLOCK_S)
        num_sms = _num_sms(x.device.index)
        if base_grid < num_sms and num_k_tiles >= 8:
            split_k = max(1, min(2 * num_sms // base_grid, num_k_tiles, 16))

    if split_k > 1:
        # Cover the whole rank in one N-block (BLOCK_N >= R => num_pid_n == 1) and
        # accumulate in a pre-zeroed fp32 buffer; cast back to x.dtype afterwards so
        # the LoRA-B expand consumer keeps its bf16 input contract unchanged.
        BLOCK_N = triton.next_power_of_2(R)
        output = torch.zeros((S, R), device=x.device, dtype=torch.float32)
    else:
        BLOCK_N = BLOCK_R
        output = torch.empty((S, R), device=x.device, dtype=x.dtype)

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_N) * split_k,
        batch_info.bs,
    )

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
        BLOCK_N,
        BLOCK_K,
        split_k,
    )
    return output.to(x.dtype) if split_k > 1 else output
