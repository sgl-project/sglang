"""WO-A split reduction with the row-major MXFP8 operand used by gfx950 WO-B."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.wo_a import GROUPS, N_OUT, RANK, _wo_a_partial
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Mxfp8Activation,
    fp8_grid_quant,
)

SPLITS = 8  # K splits of _wo_a_partial
TILE = 256  # output columns per reduce program
GRID_EPS = 1e-10  # keeps a zero tile's scale finite


@triton.jit
def _wo_a_reduce_mxfp8(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * TILE + tl.arange(0, TILE)
    split = tl.arange(0, SPLITS)
    values = tl.load(P + split[:, None] * (M * N_OUT) + row * N_OUT + i[None, :])
    # Preserve the BF16 rounding before the native WO-B activation quantizer.
    y = tl.sum(values, 0).to(tl.bfloat16).to(tl.float32).reshape((TILE // 32, 32))
    q, exponent = fp8_grid_quant(y, GRID_EPS)
    tl.store(Q + row * N_OUT + i, q.reshape((TILE,)))
    tl.store(
        S + row * (N_OUT // 32) + tile * (TILE // 32) + tl.arange(0, TILE // 32),
        exponent.to(tl.uint8),
    )


def wo_a_bf16_small_batch_mxfp8_hip(
    x: torch.Tensor, weight: torch.Tensor
) -> Mxfp8Activation:
    """Project 2 to 8 TP4 verify rows and quantize after BF16 output rounding."""
    m, hidden = x.shape[0], x.shape[-1]
    assert 2 <= m <= 8 and x.shape[1:] == (GROUPS, hidden)
    assert weight.shape == (GROUPS, RANK, hidden) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == hidden and x.stride(0) >= GROUPS * hidden
    partial = torch.empty(
        (SPLITS, m, GROUPS, RANK), dtype=torch.float32, device=x.device
    )
    q = torch.empty((m, N_OUT), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((m, N_OUT // 32), dtype=torch.uint8, device=x.device)
    _wo_a_partial[(RANK // 64, GROUPS, SPLITS)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce_mxfp8[(m, N_OUT // TILE)](partial, q, scales, m, num_warps=4)
    return Mxfp8Activation(q, scales)
