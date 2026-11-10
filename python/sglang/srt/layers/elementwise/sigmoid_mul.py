from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """Fused sigmoid-mul for attention output gating (attn_output * sigmoid(gate)).

    Supports a strided 3D gate (num_tokens, num_heads, head_dim) against a 2D
    output (num_tokens, num_heads * head_dim) as well as flat inputs.
    """
    if gate.ndim == 3 and attn_output.ndim == 2:
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        assert attn_output.shape == gate.shape, (
            "attn_output and gate must have the same shape"
        )
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out
