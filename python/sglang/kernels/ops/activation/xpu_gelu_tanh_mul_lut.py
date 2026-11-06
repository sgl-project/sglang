"""BF16 gated activation with a caller-owned native GELU value table."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lookup_mul(X, TABLE, Y, TOTAL: tl.constexpr, I: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // I, index % I
    gate = tl.load(X + row * (2 * I) + column, index < TOTAL, other=0)
    up = tl.load(X + row * (2 * I) + I + column, index < TOTAL, other=0)
    bits = gate.to(tl.uint16, bitcast=True).to(tl.uint32)
    activated = tl.load(TABLE + bits).to(tl.float32)
    tl.store(Y + index, activated * up.to(tl.float32), index < TOTAL)


def bf16_gelu_tanh_mul_lut(input: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Apply native BF16 GELU(gate) times up without recomputing GELU.

    ``input`` is a contiguous BF16 XPU matrix [M, 2*I]. The immutable BF16
    table contains all 65536 native BF16 GELU results in raw-bit order, on
    the same device. Its construction and lifetime belong to the caller.
    Multiplication uses FP32 opmath and rounds to BF16 on output. Input is
    limited to INT_MAX elements for 32-bit addressing. This is an inference
    operator; it does not compute gradients or alter GEMM.
    """
    if (
        input.ndim != 2
        or input.device.type != "xpu"
        or input.dtype != torch.bfloat16
        or not input.is_contiguous()
        or input.shape[-1] % 2
        or input.numel() > 2147483647
        or table.ndim != 1
        or table.shape[0] != 65536
        or table.device != input.device
        or table.dtype != torch.bfloat16
        or not table.is_contiguous()
        or input.requires_grad
        or table.requires_grad
    ):
        raise ValueError(
            "requires contiguous inference BF16 [M,2*I] with at most INT_MAX "
            "elements and a [65536] table on one XPU"
        )
    output = torch.empty(
        (input.shape[0], input.shape[1] // 2), device=input.device, dtype=input.dtype
    )
    if output.numel():
        _lookup_mul[(triton.cdiv(output.numel(), 1024),)](
            input, table, output, output.numel(), output.shape[-1], 1024, num_warps=4
        )
    return output
