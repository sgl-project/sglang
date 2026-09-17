"""WO-A split reduction with the row-major MXFP8 operand used by gfx950 WO-B."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.wo_a_bf16 import _wo_a_partial
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Mxfp8Activation,
    fp8_grid_quant,
)


@triton.jit
def _wo_a_reduce_mxfp8(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * (M * 2048) + row * 2048 + i[None, :])
    # Preserve the BF16 rounding before the native WO-B activation quantizer.
    y = tl.sum(values, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    q, exponent = fp8_grid_quant(y, 1e-10)
    tl.store(Q + row * 2048 + i, q.reshape((256,)))
    tl.store(S + row * 64 + tile * 8 + tl.arange(0, 8), exponent.to(tl.uint8))


def wo_a_bf16_small_batch_mxfp8_hip(
    x: torch.Tensor, weight: torch.Tensor
) -> Mxfp8Activation:
    """Project 2–8 TP4 verify rows and quantize after BF16 output rounding."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    q = torch.empty((m, 2048), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, 64), dtype=torch.uint8, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce_mxfp8[(m, 8)](partial, q, scales, m, num_warps=4)
    return Mxfp8Activation(q, scales)
