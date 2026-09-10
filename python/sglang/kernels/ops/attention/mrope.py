"""Interleaved M-RoPE Triton kernel, migrated from
``sglang.srt.layers.rotary_embedding.mrope`` (RFC #29630, Phase 2.5).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def apply_interleaved_rope_kernel(
    x_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    stride_x_m,
    stride_x_s,
    stride_out_s,
    section_1_end,
    section_2_end,
    BLOCK_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    start_s = tl.program_id(0) * BLOCK_S
    s_offsets = start_s + tl.arange(0, BLOCK_S)

    dim_offset = tl.program_id(1) * BLOCK_SIZE
    dim_indices = dim_offset + tl.arange(0, BLOCK_SIZE)

    mask_s = s_offsets < S
    mask_d = dim_indices < D
    mask = mask_s[:, None] & mask_d[None, :]

    val_ptr = (
        x_ptr + 0 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val = tl.load(val_ptr, mask=mask, other=0.0)

    cond_a = (dim_indices[None, :] % 3 == 1) & (
        dim_indices[None, :] < section_1_end * 3
    )
    val_a_ptr = (
        x_ptr + 1 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val_a = tl.load(val_a_ptr, mask=mask & cond_a, other=0.0)

    cond_b = (dim_indices[None, :] % 3 == 2) & (
        dim_indices[None, :] < section_2_end * 3
    )
    val_b_ptr = (
        x_ptr + 2 * stride_x_m + s_offsets[:, None] * stride_x_s + dim_indices[None, :]
    )
    val_b = tl.load(val_b_ptr, mask=mask & cond_b, other=0.0)

    val = tl.where(cond_a, val_a, val)
    val = tl.where(cond_b, val_b, val)

    out_ptr = out_ptr + s_offsets[:, None] * stride_out_s + dim_indices[None, :]
    tl.store(out_ptr, val, mask=mask)


def apply_interleaved_rope_triton(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x = x.contiguous()
    M, S, D = x.shape

    out = torch.empty((S, D), dtype=x.dtype, device=x.device)

    BLOCK_S = 64
    BLOCK_SIZE = 128

    grid = (triton.cdiv(S, BLOCK_S), triton.cdiv(D, BLOCK_SIZE))

    section_1_end = mrope_section[1]
    section_2_end = mrope_section[2]

    apply_interleaved_rope_kernel[grid](
        x,
        out,
        S,
        D,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        section_1_end,
        section_2_end,
        BLOCK_S=BLOCK_S,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
