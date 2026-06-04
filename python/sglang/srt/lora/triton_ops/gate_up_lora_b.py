import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.lora.triton_ops.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.utils import LoRABatchInfo

# Minimum total_tokens * rank for the single-adapter cuBLAS path; below this
# the Triton kernel is faster (crossover measured at output_dim=1536/GPU:
# cuBLAS wins rank64 from S>=256 and rank16 only from S>=2048).
_CUBLAS_MIN_S_RANK = 16384


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
    # For fused output scaling
    scalings,
    ENABLE_PDL: tl.constexpr = False,
):
    """
    This kernel packs 2 sgemms (gate/up) into a single kernel. The multiplication
    results are accumulated into the output tensor.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, 2 * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, 2 * output_dim, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, 2 * output_dim).
    """
    # output_dim >> K

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # gate_up_id decides which of gate or up (0: gate, 1: up)
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    gate_up_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = gate_up_id * output_dim  # offset on output dim
    scaling = tl.load(scalings + w_index)

    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(output_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + (gate_up_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # GDC wait: ensure the prior kernel (producer of x) has fully completed
    # before consuming its output.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

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
            mask=(k_offset[:, None] < K - k * BLOCK_K)
            & (n_offset[None, :] < output_dim),
            other=0.0,
        )
        partial_sum += tl.dot(
            x_tile.to(w_tile.dtype), w_tile
        )  # cast fused: split-K returns fp32, plain path bf16 (no-op)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # All input reads are done; hint the runtime to launch the dependent kernel.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < output_dim)
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def _gate_up_lora_b_cublas(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor,
) -> torch.Tensor:
    """Single-adapter dense path: one cuBLAS addmm_ per gate/up slice.

    The LoRA-A output is rank-packed (slice i at columns [i*rank, (i+1)*rank)),
    matching the Triton kernel's K = min(K, rank) slice stride. Slices are
    disjoint output regions, so in-place addmm_ writes never collide.
    """
    r = gate_up_lora_b.shape[-1]
    if base_output is None:
        base_output = torch.zeros(
            (x.shape[0], 2 * output_dim), device=x.device, dtype=x.dtype
        )
    w = gate_up_lora_b[0]
    x_scaled = x[:, : 2 * r] * batch_info.scalings[0]
    for i in range(2):
        lo, hi = i * output_dim, (i + 1) * output_dim
        base_output[:, lo:hi].addmm_(x_scaled[:, i * r : (i + 1) * r], w[lo:hi, :r].t())
    return base_output


def gate_up_lora_b_fwd(
    x: torch.Tensor,
    gate_up_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_dim: int,
    base_output: torch.Tensor = None,
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

    if (
        envs.SGLANG_OPT_LORA_CUBLAS.get() or envs.SGLANG_OPT_LORA_CUBLAS_GATE_UP.get()
    ) and s * r >= _CUBLAS_MIN_S_RANK:
        return _gate_up_lora_b_cublas(
            x, gate_up_lora_b, batch_info, output_dim, base_output
        )

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(output_dim, BLOCK_OUT),
        2,  # this dimension decides current block computes on gate or up proj
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, 2 * output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
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
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
        ENABLE_PDL=enable_pdl,
        **pdl_kwargs,
    )

    return output
