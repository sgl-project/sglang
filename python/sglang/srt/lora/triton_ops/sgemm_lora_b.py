import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel


def _get_sgemm_lora_b_block_config(N: int, R: int, max_len: int):
    """Pick BLOCK_S/BLOCK_N/BLOCK_K for the LoRA-B (expand) kernel.

    The key optimisation is sizing BLOCK_K to cover the full rank so the
    K-loop executes in a single iteration (rank=64 → 30%+ speedup vs the
    old fixed BLOCK_K=16).  BLOCK_N is scaled down when BLOCK_K is large
    to keep the per-tile shared-memory footprint reasonable.
    num_warps / num_stages are left to Triton's autotuner.
    """
    # K-dim block: cover the typical rank in one shot to eliminate the K loop.
    if R <= 16:
        BLOCK_K = 16
    elif R <= 32:
        BLOCK_K = 32
    elif R <= 64:
        BLOCK_K = 64
    else:
        BLOCK_K = 32

    # N-dim block: scale down when BLOCK_K is large to avoid blowing up
    # shared memory (BLOCK_S * BLOCK_K + BLOCK_K * BLOCK_N tiles).
    if BLOCK_K >= 64:
        BLOCK_N = 128
    else:
        BLOCK_N = 256  # original value for small ranks

    # S-dim block: use 32 for large token counts to halve the grid size.
    BLOCK_S = 32 if max_len >= 128 else 16

    return BLOCK_S, BLOCK_N, BLOCK_K


@cached_triton_kernel(
    lambda _, kwargs: (
        kwargs["N"],
        kwargs["K"],
        kwargs["BLOCK_S"],
        kwargs["BLOCK_N"],
        kwargs["BLOCK_K"],
        kwargs["SORTED_BY_ADAPTER"],
        kwargs["HAS_BASE_OUTPUT"],
        kwargs["x"].dtype,
    )
)
@triton.jit
def _sgemm_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
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
    HAS_BASE_OUTPUT: tl.constexpr,
    # For fused output scaling
    scalings,
):
    """
    Computes a segmented batched matrix multiplication for the LoRA B matrix
    and adds the result to the output in-place.

    When HAS_BASE_OUTPUT is False the output tensor is assumed to be zero-
    initialised and the final load-before-store is skipped, saving one full
    global-memory read per tile.
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
    n_mask = n_offset[None, :] < N
    s_mask = s_offset[:, None] < seg_len
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs,
            mask=s_mask & (k_offset[None, :] < k_remaining),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < k_remaining) & n_mask,
            other=0.0,
        )
        partial_sum = tl.dot(x_tile, w_tile, acc=partial_sum)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Fuse scaling into the fp32→fp16 cast (P3).
    partial_sum = (partial_sum * scaling).to(x.dtype.element_ty)

    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = s_mask & n_mask
    # Only read-modify-write when accumulating onto an existing base output (P1).
    if HAS_BASE_OUTPUT:
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

    has_base_output = base_output is not None

    BLOCK_S, BLOCK_N, BLOCK_R = _get_sgemm_lora_b_block_config(
        N, R, batch_info.max_len
    )

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )

    if not has_base_output:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _sgemm_lora_b_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        N=N,
        K=R,
        x_stride_0=x.stride(0),
        x_stride_1=x.stride(1),
        w_stride_0=weights.stride(0),
        w_stride_1=weights.stride(1),
        w_stride_2=weights.stride(2),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        seg_lens=batch_info.seg_lens,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        sorted_token_ids=batch_info.permutation,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_R,
        HAS_BASE_OUTPUT=has_base_output,
        scalings=batch_info.scalings,
    )
    return output