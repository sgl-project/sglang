import torch
import triton
import triton.language as tl

from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs
from sglang.kernels.ops.gemm.trtllm_lora_temp.gate_up_lora_b import (
    _CUBLAS_MIN_S_RANK,
)
from sglang.kernels.ops.gemm.trtllm_lora_temp.kernel_utils import (
    _resolve_token_positions,
    get_pdl_launch_metadata,
)
from sglang.srt.lora.utils import LoRABatchInfo


def _sgemm_lora_b_cublas(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
) -> torch.Tensor:
    """Single-adapter dense path: one cuBLAS addmm_ over the full output.

    Mirrors the Triton kernel exactly: single slice, K = max_r (the kernel
    reads the full K; the loader zero-pads the weight tail beyond the
    adapter's rank), scaling fused via a pre-scaled x.
    """
    if base_output is None:
        base_output = torch.zeros(
            (x.shape[0], weights.shape[-2]), device=x.device, dtype=x.dtype
        )
    w = weights[batch_info.weight_indices[0]]
    x_scaled = x * batch_info.scalings[0]
    base_output.addmm_(x_scaled, w.t())
    return base_output


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
    # For fused output scaling
    scalings,
    ENABLE_PDL: tl.constexpr = False,
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

    pid_s = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    seg_len = tl.load(seg_lens + batch_id)
    if pid_s * BLOCK_S >= seg_len:  # also covers seg_len == 0
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)

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

    # GDC wait: ensure the prior kernel (producer of x) has fully completed
    # before consuming its output.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < N
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask

    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(
        w_ptrs,
        mask=(k_offset[:, None] < K) & n_mask,
        other=0.0,
    )

    # cast fused: the split-K shrink returns fp32, plain path bf16 (no-op)
    partial_sum = tl.dot(x_tile.to(w_tile.dtype), w_tile) * scaling

    # All input reads are done; hint the runtime to launch the dependent kernel.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Store result to output matrix (cast to the OUTPUT dtype: x may be the fp32
    # split-K shrink accumulator while base_output is bf16)
    partial_sum = partial_sum.to(output.dtype.element_ty)
    tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")


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

    if (
        (
            lora_envs.SGLANG_OPT_LORA_CUBLAS.get()
            or lora_envs.SGLANG_OPT_LORA_CUBLAS_B.get()
        )
        and S * R >= _CUBLAS_MIN_S_RANK
        and weights.shape[0] == 1
    ):  # single-adapter fast path: only valid with one resident slot
        return _sgemm_lora_b_cublas(x, weights, batch_info, base_output)
    # Block shapes
    BLOCK_S = 16
    BLOCK_R = triton.next_power_of_2(R)
    BLOCK_N = 256

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S),
        triton.cdiv(N, BLOCK_N),
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((S, N), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _sgemm_lora_b_kernel[grid](
        x,
        weights,
        output,
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
        ENABLE_PDL=enable_pdl,
        **pdl_kwargs,
    )
    return output
