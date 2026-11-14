"""Fused two-dimensional RoPE for the DeepSeek-V4.1 vision encoder.

This narrow CUDA path expects 64-dimensional heads and float32 cosine/sine
tables. The caller validates the layout and falls back for unsupported inputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_rope_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    q_token_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    num_heads: tl.constexpr,
    heads_per_program: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, heads_per_program * 32)
    head = group * heads_per_program + offsets // 32
    dim = offsets % 32
    valid_head = head < num_heads

    cosine = tl.load(cos_ptr + token * 32 + dim)
    sine = tl.load(sin_ptr + token * 32 + dim)
    q_first = tl.load(
        q_ptr + token * q_token_stride + head * q_head_stride + dim, valid_head, 0
    ).to(tl.float32)
    q_second = tl.load(
        q_ptr + token * q_token_stride + head * q_head_stride + 32 + dim, valid_head, 0
    ).to(tl.float32)
    k_first = tl.load(
        k_ptr + token * k_token_stride + head * k_head_stride + dim, valid_head, 0
    ).to(tl.float32)
    k_second = tl.load(
        k_ptr + token * k_token_stride + head * k_head_stride + 32 + dim, valid_head, 0
    ).to(tl.float32)

    output = (token * num_heads + head) * 64
    tl.store(q_out_ptr + output + dim, q_first * cosine - q_second * sine, valid_head)
    tl.store(
        q_out_ptr + output + 32 + dim, q_second * cosine + q_first * sine, valid_head
    )
    tl.store(k_out_ptr + output + dim, k_first * cosine - k_second * sine, valid_head)
    tl.store(
        k_out_ptr + output + 32 + dim, k_second * cosine + k_first * sine, valid_head
    )


def apply_fused_vision_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, _ = q.shape
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    heads_per_program = 8
    _vision_rope_qk_kernel[(num_tokens, triton.cdiv(num_heads, heads_per_program))](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        num_heads,
        heads_per_program,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, k_out
