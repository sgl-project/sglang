"""Two BF16 pointwise operations with their eager intermediate rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_add_then_scale(X, Y, SCALE, OUT, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < TOTAL, other=0).to(tl.float32)
    y = tl.load(Y + offsets, offsets < TOTAL, other=0).to(tl.float32)
    scale = tl.load(SCALE).to(tl.float32)
    # This conversion is part of the function, not an implementation detail:
    # eager BF16 add stores before the following scalar multiply reads it.
    added = (x + y).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + offsets, added * scale, offsets < TOTAL)


def bf16_add_then_scale(
    x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Return ``(x + y) * scale`` with a BF16 add before the multiply.

    Both inputs are contiguous BF16 matrices on one XPU. ``scale`` is a
    contiguous BF16 tensor of shape ``[1]`` on that device. Inputs are limited
    to INT_MAX elements for 32-bit addressing. The returned BF16 tensor owns
    new storage, and all operands are read again on graph replay. This
    inference operator does not compute gradients.
    """
    if (
        x.ndim != 2
        or x.device.type != "xpu"
        or x.dtype != torch.bfloat16
        or y.shape != x.shape
        or y.device != x.device
        or y.dtype != torch.bfloat16
        or scale.shape != (1,)
        or scale.device != x.device
        or scale.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not y.is_contiguous()
        or not scale.is_contiguous()
        or x.numel() > 2147483647
        or x.requires_grad
        or y.requires_grad
        or scale.requires_grad
    ):
        raise ValueError(
            "requires contiguous inference BF16 matrices with at most INT_MAX "
            "elements and a BF16 [1] scalar on one XPU"
        )
    output = torch.empty_like(x)
    if output.numel():
        _bf16_add_then_scale[(triton.cdiv(output.numel(), 1024),)](
            x, y, scale, output, output.numel(), 1024, num_warps=4
        )
    return output
