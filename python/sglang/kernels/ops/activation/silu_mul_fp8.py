# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded SwiGLU with dynamic per-row E4M3 quantization."""

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import div_rn_f32, round_bf16_to_fp32


@triton.jit
def _silu_mul_fp8_kernel(G, U, Q, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    gate = tl.load(G + row * N + col, col < N, 0).to(tl.float32)
    up = tl.load(U + row * N + col, col < N, 0).to(tl.float32)
    # Retain both BF16 stores from eager silu(gate) * up.
    activation = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    hidden = round_bf16_to_fp32(activation * up)
    amax = tl.max(tl.abs(hidden), 0)
    scale = div_rn_f32(amax, 448.0)
    inverse = tl.where(scale == 0.0, 0.0, div_rn_f32(1.0, scale))
    value = tl.minimum(tl.maximum(hidden * inverse, -448.0), 448.0)
    # SM89's default Triton cast can pass through FP16, double-rounding
    # values near E4M3 halfway points. Convert directly from FP32 instead.
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 packed; cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1; cvt.u32.u16 $0, packed; }",
        constraints="=r,f",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    quantized = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    tl.store(Q + row * N + col, quantized, col < N)
    tl.store(S + row, scale)


def silu_mul_fp8(gate: torch.Tensor, up: torch.Tensor):
    """Return E4M3 [M,N] and FP32 [M,1] row scales on CUDA SM89+.

    Inputs are contiguous BF16 matrices, with 0 < N <= 16384. The producer
    retains BF16 rounding after SiLU and multiplication, then uses correctly
    rounded FP32 amax/448 and its reciprocal. Zero rows have zero scale and
    payload. Inputs and BF16 intermediate results must be finite.
    """
    if not (
        gate.is_cuda
        and torch.version.hip is None
        and gate.dtype is torch.bfloat16
        and up.dtype is torch.bfloat16
        and gate.device == up.device
        and gate.ndim == 2
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and 0 < gate.shape[1] <= 16384
        and torch.cuda.get_device_capability(gate.device) >= (8, 9)
    ):
        raise ValueError(
            "silu_mul_fp8 requires equal contiguous BF16 matrices on SM89+, N in [1, 16384]"
        )
    rows, hidden = gate.shape
    quantized = torch.empty_like(gate, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), device=gate.device, dtype=torch.float32)
    if rows:
        with torch.cuda.device(gate.device):
            _silu_mul_fp8_kernel[(rows,)](
                gate,
                up,
                quantized,
                scales,
                hidden,
                triton.next_power_of_2(hidden),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return quantized, scales
