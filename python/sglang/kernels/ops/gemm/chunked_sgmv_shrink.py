import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_shrink_config
from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel


@cached_triton_kernel(
    lambda _, kwargs: (kwargs["K"], kwargs["NUM_SLICES"], kwargs["BLOCK_M"])
)
@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA shrink operations.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_split_k_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
):
    """Write FP32 partials with shape [SPLIT_K, TOTAL_ROWS, N]."""
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    if seg_start == seg_end:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end, other=0
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    split_id = tl.program_id(2)
    steps: tl.constexpr = tl.cdiv(K, BLOCK_K * SPLIT_K)
    k_offset = tl.arange(0, BLOCK_K) + split_id * steps * BLOCK_K
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, steps):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    output_ptr = (
        output
        + split_id * TOTAL_ROWS * N
        + (
            s_offset_physical[:, None] * output_stride_0
            + n_offset[None, :] * output_stride_1
        )
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


@triton.jit
def _reduce_shrink_split_k(
    partials,
    output,
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    N: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    segment = tl.program_id(1)
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    if start == end:
        return
    adapter = tl.load(weight_indices + segment)
    rank = tl.load(lora_ranks + adapter)
    if rank == 0:
        return
    logical_rows = start + tl.arange(0, BLOCK_M)
    rows = tl.load(permutation + logical_rows, mask=logical_rows < end, other=0)
    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (logical_rows[:, None] < end) & (
        columns[None, :] < tl.minimum(N, rank * NUM_SLICES)
    )
    offsets = rows[:, None] * N + columns[None, :]
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(partials + split * TOTAL_ROWS * N + offsets, mask=mask, other=0)
    tl.store(output + offsets, acc, mask=mask)


def chunked_sgmv_lora_shrink_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int,
    split_k: int = 1,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, num_slices * r, input_dim)
    # output: (s, num_slices * r)
    # num_slices: qkv=3, gate_up=2, others=1
    # when called with multiple slices, the weights.shape[-2] will be num_slices * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Block shapes — use auto-tuned config if available, else defaults
    BLOCK_M = batch_info.max_len
    # weights shape is (num_lora, num_slices * rank, input_dim)
    MAX_RANK = weights.shape[1] // num_slices
    config = get_lora_shrink_config(
        K=weights.shape[2], R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M
    )
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K

    num_segments = batch_info.num_segments
    segment_grid = (
        batch_info.weight_indices.shape[0]
        if batch_info.use_cuda_graph
        else num_segments
    )
    grid = (
        triton.cdiv(N, BLOCK_N),
        segment_grid,
    )

    # Optional launch params from tuned config
    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]

    # Opt-in policy measured on H200 with BF16 rank-32 adapters. Keep other
    # shapes and deterministic inference on the original accumulation order.
    if (
        envs.SGLANG_CSGMV_SPLIT_K.get()
        and MAX_RANK == 32
        and K in (4096, 12288)
        and x.dtype == torch.bfloat16
        and (BLOCK_M, BLOCK_N, BLOCK_K) == (16, 16, 256)
    ):
        split_k = 16 if K == 12288 else 8
    if envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        split_k = 1
    if split_k < 1:
        raise ValueError("split_k must be positive")

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    if split_k > 1:
        partials = torch.empty((split_k, S, N), device=x.device, dtype=torch.float32)
        _chunked_lora_shrink_split_k_kernel[(*grid, split_k)](
            x=x,
            weights=weights,
            output=partials,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=segment_grid,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=split_k,
            TOTAL_ROWS=S,
            **extra_kwargs,
        )
        # Reduce only live rows/rank columns. Unused adapter slots and graph
        # padding have no initialized partials and must not be read.
        _reduce_shrink_split_k[grid](
            partials,
            output,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.permutation,
            N=N,
            TOTAL_ROWS=S,
            NUM_SLICES=num_slices,
            SPLIT_K=split_k,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return output
    _chunked_lora_shrink_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=segment_grid,
        # constants
        N=N,
        K=K,
        NUM_SLICES=num_slices,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **extra_kwargs,
    )

    return output
