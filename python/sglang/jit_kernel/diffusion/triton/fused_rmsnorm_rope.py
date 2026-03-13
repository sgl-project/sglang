"""Fused RMSNorm + Interleaved RoPE kernel for MOVA DiT.

This module provides a fused kernel that combines RMSNorm normalization
with interleaved RoPE (Rotary Position Embedding) in a single kernel launch.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _fused_rmsnorm_rope_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    M,  # total rows (B * S)
    D: tl.constexpr,  # hidden dimension
    seq_len,  # sequence length (for cos/sin indexing)
    head_dim_half: tl.constexpr,  # head_dim // 2
    eps,
    stride_x_row,
    stride_out_row,
    stride_cos_row,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm + interleaved RoPE kernel.

    Two-pass algorithm:
      Pass 1: accumulate sum-of-squares over the row to compute rstd.
      Pass 2: normalize with weight, then apply interleaved RoPE in pairs.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * stride_x_row
    out_row_ptr = out_ptr + row * stride_out_row

    # Token index within the sequence (for cos/sin lookup)
    token_idx = row % seq_len

    # --- Pass 1: compute rstd = rsqrt(mean(x^2) + eps) ---
    sum_sq = tl.zeros([], dtype=tl.float32)
    for block_start in tl.range(0, D, BLOCK_D):
        offsets = block_start + tl.arange(0, BLOCK_D)
        mask = offsets < D
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    rstd = tl.math.rsqrt(sum_sq / D + eps)

    # --- Pass 2: fused normalize + RoPE (process in pairs) ---
    # Access even/odd elements directly from global memory with stride-2 loads.
    # Global memory stride-2 loads do NOT go through shared memory, so they
    # do not cause shared memory bank conflicts.
    num_pairs: tl.constexpr = D // 2
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2

    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_cos_row

    for block_start in tl.range(0, num_pairs, BLOCK_PAIRS):
        pair_offsets = block_start + tl.arange(0, BLOCK_PAIRS)
        pair_mask = pair_offsets < num_pairs

        # Stride-2 global memory loads: go through L2/L1 cache, NOT shared mem
        even_offsets = pair_offsets * 2
        odd_offsets = pair_offsets * 2 + 1

        x_even = tl.load(x_row_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        x_odd = tl.load(x_row_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        w_even = tl.load(weight_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        w_odd = tl.load(weight_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        # Normalize
        xn_even = x_even * rstd * w_even
        xn_odd = x_odd * rstd * w_odd

        # RoPE index: wraps every head_dim_half pairs
        rope_idx = pair_offsets % head_dim_half

        # cos/sin: contiguous access (rope_idx is contiguous)
        cos_vals = tl.load(cos_row_ptr + rope_idx, mask=pair_mask, other=1.0).to(
            tl.float32
        )
        sin_vals = tl.load(sin_row_ptr + rope_idx, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        # Interleaved RoPE
        o_even = xn_even * cos_vals - xn_odd * sin_vals
        o_odd = xn_even * sin_vals + xn_odd * cos_vals

        # Store back at original stride-2 positions
        tl.store(out_row_ptr + even_offsets, o_even, mask=pair_mask)
        tl.store(out_row_ptr + odd_offsets, o_odd, mask=pair_mask)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + interleaved RoPE.

    Args:
        x: Input tensor [B, S, D] or [*, D] (bf16/fp16).
        weight: RMSNorm weight [D] (any dtype, cast to fp32 internally).
        cos: Precomputed cosine [S, head_dim//2] float32.
        sin: Precomputed sine [S, head_dim//2] float32.
        head_dim: Per-head dimension (e.g. 128).
        eps: RMSNorm epsilon.

    Returns:
        Output tensor, same shape and dtype as x.
    """
    # --- Precondition checks ---
    assert x.is_cuda, "x must be on CUDA"
    assert weight.is_cuda, "weight must be on CUDA"
    assert cos.is_cuda, "cos must be on CUDA"
    assert sin.is_cuda, "sin must be on CUDA"

    orig_shape = x.shape
    seq_len = cos.shape[0]
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, D = x_2d.shape
    assert D % 2 == 0, f"Hidden dim must be even, got {D}"

    out = torch.empty_like(x_2d)
    head_dim_half = head_dim // 2

    grid = (M,)
    _fused_rmsnorm_rope_kernel[grid](
        out,
        x_2d,
        weight,
        cos,
        sin,
        M,
        D,
        seq_len,
        head_dim_half,
        eps,
        x_2d.stride(0),
        out.stride(0),
        cos.stride(0),
    )
    return out.view(orig_shape)
