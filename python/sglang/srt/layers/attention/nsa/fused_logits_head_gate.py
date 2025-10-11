"""
Triton fused kernel for logits head gate computation in NSA indexer.

This kernel fuses the following operations:
    weights = weights * n_heads**-0.5
    weights = weights.unsqueeze(-1) * q_scale * softmax_scale

into a single optimized kernel.

The computation is performed in float32 for numerical stability and outputs
float32 regardless of input dtype (bf16/fp16/fp32).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_logits_head_gate_kernel(
    weights_ptr,  # Input: (M, N) where M is batch/seq_len, N is n_heads
    q_scale_ptr,  # Input: (M, N, 1) quantization scale
    output_ptr,  # Output: (M, N, 1)
    n_heads_inv_sqrt: tl.constexpr,  # 1 / sqrt(n_heads)
    softmax_scale: tl.constexpr,  # head_dim**-0.5
    M,  # Batch size / sequence length
    N,  # Number of heads
    weights_stride_m,
    weights_stride_n,
    q_scale_stride_m,
    q_scale_stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for computing logits head gate.

    Performs: output[m, n, 0] = weights[m, n] * n_heads**-0.5 * q_scale[m, n, 0] * softmax_scale
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    weights_ptrs = (
        weights_ptr
        + offs_m[:, None] * weights_stride_m
        + offs_n[None, :] * weights_stride_n
    )
    q_scale_ptrs = (
        q_scale_ptr
        + offs_m[:, None] * q_scale_stride_m
        + offs_n[None, :] * q_scale_stride_n
    )
    output_ptrs = (
        output_ptr
        + offs_m[:, None] * output_stride_m
        + offs_n[None, :] * output_stride_n
    )

    weights = tl.load(weights_ptrs, mask=mask, other=0.0).to(tl.float32)
    q_scale = tl.load(q_scale_ptrs, mask=mask, other=0.0).to(tl.float32)

    result = weights * n_heads_inv_sqrt * q_scale * softmax_scale
    tl.store(output_ptrs, result, mask=mask)


def fused_logits_head_gate(
    weights: torch.Tensor,  # (M, N)
    q_scale: torch.Tensor,  # (M, N, 1)
    n_heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Fused implementation of logits head gate computation.

    Computes: output = weights * n_heads**-0.5 * q_scale * softmax_scale

    The computation is performed in float32 for numerical stability,
    regardless of input dtype.

    Args:
        weights: Input weights tensor of shape (M, N) where M is batch/seq_len
                 and N is number of heads. Can be any dtype (bf16, fp16, fp32).
        q_scale: Quantization scale tensor of shape (M, N, 1). Can be any dtype.
        n_heads: Number of attention heads
        softmax_scale: Softmax scale factor (typically head_dim**-0.5)

    Returns:
        Output tensor of shape (M, N, 1) in float32 dtype
    """
    assert weights.ndim == 2, f"weights must be 2D, got {weights.ndim}D"
    assert q_scale.ndim == 3, f"q_scale must be 3D, got {q_scale.ndim}D"
    assert q_scale.shape[2] == 1, f"q_scale last dim must be 1, got {q_scale.shape[2]}"
    assert weights.shape[0] == q_scale.shape[0], "Batch dimension mismatch"
    assert weights.shape[1] == q_scale.shape[1], "Number of heads mismatch"
    assert weights.is_contiguous(), "weights must be contiguous"

    M, N = weights.shape

    output = torch.empty((M, N, 1), dtype=torch.float32, device=weights.device)

    n_heads_inv_sqrt = n_heads**-0.5

    BLOCK_SIZE_M = min(32, triton.next_power_of_2(M))
    BLOCK_SIZE_N = min(64, triton.next_power_of_2(N))

    # Launch kernel
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    fused_logits_head_gate_kernel[grid](
        weights,
        q_scale,
        output,
        n_heads_inv_sqrt,
        softmax_scale,
        M,
        N,
        weights.stride(0),
        weights.stride(1),
        q_scale.stride(0),
        q_scale.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output
