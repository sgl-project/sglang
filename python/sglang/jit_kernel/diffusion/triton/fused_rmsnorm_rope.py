"""Fused RMSNorm + Interleaved RoPE kernel for MOVA DiT.

This module provides a fused kernel that combines RMSNorm normalization
with interleaved RoPE (Rotary Position Embedding) in a single kernel launch.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 1}, num_warps=8),
        triton.Config({"ROWS_PER_CTA": 2}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 2}, num_warps=8),
        triton.Config({"ROWS_PER_CTA": 4}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 4}, num_warps=8),
        triton.Config({"ROWS_PER_CTA": 8}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 8}, num_warps=8),
        triton.Config({"ROWS_PER_CTA": 16}, num_warps=4),
        triton.Config({"ROWS_PER_CTA": 16}, num_warps=8),
    ],
    key=["D", "M"],
)
@triton.jit
def _fused_rmsnorm_rope_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    M,  # total rows (B * S)
    D,  # hidden dimension (runtime value)
    BLOCK_D: tl.constexpr,  # next power of two >= D; entire row fits in one block
    seq_len,  # sequence length (for cos/sin indexing)
    head_dim_half: tl.constexpr,  # head_dim // 2
    eps,
    stride_x_row,
    stride_out_row,
    stride_cos_row,
    ROWS_PER_CTA: tl.constexpr,  # rows each CTA handles; increases grid fill
):
    """Fused RMSNorm + interleaved RoPE kernel.

    Each CTA processes ROWS_PER_CTA consecutive rows so the grid size is
    ceil(M / ROWS_PER_CTA), allowing the kernel to fill more SMs when M is
    small (e.g. short sequences or small batches).

    Algorithm per row:
      Pass 1: contiguous load of x, accumulate sum-of-squares for rstd.
      Pass 2: stride-2 global loads for even/odd elements of x and weight.
              Global loads hit L1/L2 cache (84%+ hit rate) and bypass SMEM
              entirely — no shared memory bank conflicts.
              Interleaved RoPE applied per pair; stride-2 stores back.
    """
    cta_id = tl.program_id(0)
    row_start = cta_id * ROWS_PER_CTA

    # Pre-compute weight even/odd offsets (shared across all rows in this CTA)
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2
    pair_offsets = tl.arange(0, BLOCK_PAIRS)
    even_offsets = pair_offsets * 2
    odd_offsets = pair_offsets * 2 + 1

    # Load weight once per CTA (same for all rows)
    num_pairs = D // 2
    pair_mask = pair_offsets < num_pairs
    w_even = tl.load(weight_ptr + even_offsets, mask=pair_mask, other=0.0).to(
        tl.float32
    )
    w_odd = tl.load(weight_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    rope_idx = pair_offsets % head_dim_half

    for i in tl.static_range(ROWS_PER_CTA):
        row = row_start + i
        if row < M:
            x_row_ptr = x_ptr + row * stride_x_row
            out_row_ptr = out_ptr + row * stride_out_row
            token_idx = row % seq_len

            # --- Pass 1: compute rstd ---
            x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_sq = tl.sum(x_vals * x_vals, axis=0)
            rstd = tl.math.rsqrt(sum_sq / D + eps)

            # --- Pass 2: normalize + RoPE ---
            x_even = tl.load(x_row_ptr + even_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )
            x_odd = tl.load(x_row_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            xn_even = x_even * rstd * w_even
            xn_odd = x_odd * rstd * w_odd

            cos_row_ptr = cos_ptr + token_idx * stride_cos_row
            sin_row_ptr = sin_ptr + token_idx * stride_cos_row
            cos_vals = tl.load(cos_row_ptr + rope_idx, mask=pair_mask, other=1.0).to(
                tl.float32
            )
            sin_vals = tl.load(sin_row_ptr + rope_idx, mask=pair_mask, other=0.0).to(
                tl.float32
            )

            o_even = xn_even * cos_vals - xn_odd * sin_vals
            o_odd = xn_even * sin_vals + xn_odd * cos_vals

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

    # BLOCK_D must be a power of two >= D (Triton tl.arange constraint).
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    grid = lambda meta: (triton.cdiv(M, meta["ROWS_PER_CTA"]),)
    _fused_rmsnorm_rope_kernel[grid](
        out,
        x_2d,
        weight,
        cos,
        sin,
        M,
        D,
        BLOCK_D,
        seq_len,
        head_dim_half,
        eps,
        x_2d.stride(0),
        out.stride(0),
        cos.stride(0),
    )
    return out.view(orig_shape)
