# SPDX-License-Identifier: Apache-2.0
"""Packed BF16 SwiGLU directly to row-scaled E4M3, without a BF16 output."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32


@triton.jit
def _swiglu_fp8_rowwise_kernel(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = (row < M) & (col < K)
    gate = tl.load(X + row * STRIDE + col, mask, other=0).to(tl.float32)
    up = tl.load(X + row * STRIDE + K + col, mask, other=0).to(tl.float32)
    # Reproduce both BF16 rounding boundaries of F.silu(gate) * up.
    silu = round_bf16_to_fp32(gate * tl.sigmoid(gate))
    value = round_bf16_to_fp32(silu * up)
    scale = tl.maximum(tl.max(tl.abs(value), 0) * (1.0 / 448.0), 1.0e-12)
    normalized = tl.div_rn(value, scale)
    normalized = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 packed; cvt.rn.satfinite.e4m3x2.f32 packed, $1, $1; cvt.u32.u16 $0, packed; }",
        constraints="=r,f",
        args=[normalized],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    quantized = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    tl.store(Q + row * K + col, quantized, col < K)
    tl.store(S + row, scale)


@triton.jit
def _swiglu_fp8_rowwise_tiled_kernel(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, TILE)
    values = ()
    maxima = tl.full((TILE,), 0, tl.float32)
    # Keep each tile's BF16-rounded activation in registers, but reduce the
    # elementwise maxima first: only one cross-warp reduction per row.
    for tile in tl.static_range(triton.cdiv(K, TILE)):
        col = columns + tile * TILE
        mask = (row < M) & (col < K)
        gate = tl.load(X + row * STRIDE + col, mask, other=0).to(tl.float32)
        up = tl.load(X + row * STRIDE + K + col, mask, other=0).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        value = (silu * up).to(tl.bfloat16).to(tl.float32)
        values += (value,)
        maxima = tl.maximum(maxima, tl.abs(value))
    scale = tl.maximum(tl.max(maxima, 0) * (1.0 / 448.0), 1.0e-12)
    for tile in tl.static_range(triton.cdiv(K, TILE)):
        col = columns + tile * TILE
        normalized = tl.div_rn(values[tile], scale)
        normalized = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
        # Convert two distinct FP32 elements per instruction; PTX places the
        # second source in the low byte. Keep the direct FP32->E4M3 rounding.
        bits = tl.inline_asm_elementwise(
            "{ .reg .b16 packed; cvt.rn.satfinite.e4m3x2.f32 packed, $3, $2; "
            "cvt.u32.u16 $0, packed; shr.b32 $1, $0, 8; }",
            constraints="=r,=r,f,f",
            args=[normalized],
            dtype=tl.int32,
            is_pure=True,
            pack=2,
        )
        quantized = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        tl.store(Q + row * K + col, quantized, col < K)
    tl.store(S + row, scale)


def fused_packed_swiglu_fp8_rowwise(x: torch.Tensor, row_alignment: int = 16):
    """Quantize ``F.silu(gate) * up`` from packed BF16 ``[B, T, 2*K]``.

    Return contiguous E4M3 ``[ceil(B*T/alignment)*alignment, K]`` plus
    FP32 row scales. Accept row-strided slices of a packed QKV/MLP projection
    when the B/T dimensions can be flattened without a copy. Finite inputs
    reproduce both intermediate BF16 rounds and the existing rowwise FP8
    contract. Padding rows have zero payload and scale 1e-12. NVIDIA SM89+.
    Used by the FLUX3 FP8r MLP output projection.
    """
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and x.stride(-1) == 1
        and x.shape[-1] % 2 == 0
        and 0 < x.shape[-1] // 2 <= 16384
        and x.stride(1) >= x.shape[-1]
        and (x.shape[0] <= 1 or x.stride(0) == x.shape[1] * x.stride(1))
        and torch.cuda.get_device_capability(x.device) >= (8, 9)
    ):
        raise ValueError(
            "requires row-strided packed BF16 [B,T,2*K], K in [1,16384], on SM89+"
        )
    if (
        not isinstance(row_alignment, int)
        or isinstance(row_alignment, bool)
        or row_alignment <= 0
    ):
        raise ValueError("row_alignment must be a positive integer")
    rows = x.shape[0] * x.shape[1]
    hidden = x.shape[-1] // 2
    padded_rows = triton.cdiv(rows, row_alignment) * row_alignment
    q = torch.empty((padded_rows, hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((padded_rows,), device=x.device, dtype=torch.float32)
    if padded_rows:
        num_warps = 4 if hidden <= 4096 else 8
        # SM89 measurements: extra warps help underfilled grids; eight win
        # on the model's large 3072/9216-wide rows without register spills.
        if torch.cuda.get_device_capability(x.device) == (8, 9) and hidden in (
            3072,
            9216,
        ):
            num_warps = 16 if padded_rows <= 64 else 8
        with torch.cuda.device(x.device):
            if torch.cuda.get_device_capability(x.device) == (8, 9) and hidden == 9216:
                # Nine exact-width tiles instead of a 16384-wide row; measured
                # on SM89, with unchanged BF16 and E4M3 rounding boundaries.
                _swiglu_fp8_rowwise_tiled_kernel[(padded_rows,)](
                    x,
                    q,
                    scales,
                    rows,
                    hidden,
                    x.stride(1),
                    1024,
                    num_warps=16,
                    enable_fp_fusion=False,
                )
                return q, scales
            _swiglu_fp8_rowwise_kernel[(padded_rows,)](
                x,
                q,
                scales,
                rows,
                hidden,
                x.stride(1),
                triton.next_power_of_2(hidden),
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
    return q, scales
