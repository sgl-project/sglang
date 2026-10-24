"""Per-head attention gate ``out * (2 * sigmoid(gate_logits))`` for LTX-2.3.

Eager broadcasts a ``[B, T, H, 1]`` gate over ``[B, T, H, D]``, which PyTorch
runs as a non-vectorized strided kernel, plus a separate ``2 *`` pass. This
kernel takes the eager ``sigmoid(gate_logits)`` and applies the doubling and
the product in fp32 with one bf16 rounding, the same operations and rounding
as eager, so the output is bit-identical.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _head_gate_mul_kernel(
    out_ptr,
    x_ptr,
    gate_ptr,
    hidden: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * num_heads + cols // head_dim, mask=mask, other=0.0
    ).to(tl.float32)
    # eager rounds 2 * sigmoid to bf16 first; doubling a bf16 value is exact
    y = x * (gate * 2.0)
    tl.store(out_ptr + row * hidden + cols, y.to(tl.bfloat16), mask=mask)


def can_use_head_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """``x``: contiguous ``[B, T, H, D]`` or ``[B, T, H * D]``; ``gate``: ``[B, T, H]``."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.device == gate.device
        and x.is_contiguous()
        and gate.is_contiguous()
        and x.dim() in (3, 4)
        and gate.dim() == 3
        and x.shape[:2] == gate.shape[:2]
        and math.prod(x.shape[2:]) % gate.shape[-1] == 0
        and math.prod(x.shape[2:]) <= 16384
    )


def head_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x * (2 * gate)`` per head, returned as ``[B, T, H * D]``."""
    x = x.view(x.shape[0], x.shape[1], -1)
    hidden = int(x.shape[-1])
    num_heads = int(gate.shape[-1])
    out = torch.empty_like(x)
    rows = int(x.shape[0] * x.shape[1])
    if rows:
        _head_gate_mul_kernel[(rows,)](
            out,
            x,
            gate,
            hidden,
            num_heads,
            hidden // num_heads,
            BLOCK_N=triton.next_power_of_2(hidden),
            num_warps=4 if hidden <= 4096 else 8,
        )
    return out
