# SPDX-License-Identifier: MIT
# Ported from https://github.com/efsotr/nvfp4quant_test (MIT, Copyright (c) 2026
# Li Lin) for sgl-project/sglang#27246. ScaleSweep MSE NVFP4 quantization.

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_SIZE = 16
LOWER_BOUND = -3
UPPER_BOUND = 7
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 256.0
REF_MAX_SCALE_RAW = 126


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def create_fp4_scale_tensor(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> torch.Tensor:
    if is_sf_swizzled_layout:
        rounded_m = round_up(m, 128)
        rounded_n = round_up(n // BLOCK_SIZE, 4)
        return torch.empty((rounded_m, rounded_n), device=device, dtype=torch.float8_e4m3fn)
    return torch.empty((m, n // BLOCK_SIZE), device=device, dtype=torch.float8_e4m3fn)


def create_fp4_output_tensors(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    output_scale = create_fp4_scale_tensor(m, n, device, is_sf_swizzled_layout)
    return output, output_scale


@triton.jit
def swizzled_scale_offsets(
    block_offsets,
    BLOCKS_PER_OUT: tl.constexpr,
    BLOCKS_PER_OUT_PAD: tl.constexpr,
):
    row = block_offsets // BLOCKS_PER_OUT
    col = block_offsets % BLOCKS_PER_OUT

    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row_in_tile & 31

    major_k = col >> 2
    inner_k = col & 3

    return major_m * (BLOCKS_PER_OUT_PAD * 128) + major_k * 512 + inner_m * 16 + tile_m * 4 + inner_k


@triton.jit
def fp32_round_to_fp4_code(x):
    ax = tl.abs(x)
    le2 = ax <= 2.0
    le4 = ax <= 4.0
    exp = tl.where(le2, 0.5, tl.where(le4, 1.0, 2.0))
    r = libdevice.round(ax / exp)
    mag = tl.where(le2, r, tl.where(le4, r + 2.0, tl.minimum(r + 4.0, 7.0))).to(tl.uint8)
    sign = (x < 0.0).to(tl.uint8) << 3
    return mag | sign


@triton.jit
def fp32_round_to_fp4_value(x):
    ax = tl.abs(x)
    exp = tl.where(ax <= 2.0, 0.5, tl.where(ax <= 4.0, 1.0, 2.0))
    q = libdevice.round(x / exp) * exp
    return tl.minimum(tl.maximum(q, -6.0), 6.0)


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    b0 = (fp32_round_to_fp4_code(x0) | (fp32_round_to_fp4_code(x1) << 4)).to(tl.uint32)
    b1 = (fp32_round_to_fp4_code(x2) | (fp32_round_to_fp4_code(x3) << 4)).to(tl.uint32)
    b2 = (fp32_round_to_fp4_code(x4) | (fp32_round_to_fp4_code(x5) << 4)).to(tl.uint32)
    b3 = (fp32_round_to_fp4_code(x6) | (fp32_round_to_fp4_code(x7) << 4)).to(tl.uint32)
    b4 = (fp32_round_to_fp4_code(x8) | (fp32_round_to_fp4_code(x9) << 4)).to(tl.uint32)
    b5 = (fp32_round_to_fp4_code(x10) | (fp32_round_to_fp4_code(x11) << 4)).to(tl.uint32)
    b6 = (fp32_round_to_fp4_code(x12) | (fp32_round_to_fp4_code(x13) << 4)).to(tl.uint32)
    b7 = (fp32_round_to_fp4_code(x14) | (fp32_round_to_fp4_code(x15) << 4)).to(tl.uint32)

    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.u8.u32 b0, $2;
          cvt.u8.u32 b1, $3;
          cvt.u8.u32 b2, $4;
          cvt.u8.u32 b3, $5;
          cvt.u8.u32 b4, $6;
          cvt.u8.u32 b5, $7;
          cvt.u8.u32 b6, $8;
          cvt.u8.u32 b7, $9;

          mov.b32 $0, {b0, b1, b2, b3};
          mov.b32 $1, {b4, b5, b6, b7};
        }
        """,
        constraints="=r,=r,r,r,r,r,r,r,r,r",
        args=[b0, b1, b2, b3, b4, b5, b6, b7],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return lo, hi


@triton.jit
def _fp32x2_e2m1_quant_squared_error(x0, x1):
    q0 = fp32_round_to_fp4_value(x0)
    q1 = fp32_round_to_fp4_value(x1)
    return tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .f32 d0;
          .reg .f32 d1;

          sub.rn.f32 d0, $3, $1;
          sub.rn.f32 d1, $4, $2;
          mul.rn.f32 d0, d0, d0;
          fma.rn.f32 $0, d1, d1, d0;
        }
        """,
        constraints="=f,f,f,f,f",
        args=[x0, x1, q0, q1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32x2_e2m1_quant_squared_error_acc(acc, x0, x1):
    q0 = fp32_round_to_fp4_value(x0)
    q1 = fp32_round_to_fp4_value(x1)
    return tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .f32 d0;
          .reg .f32 d1;

          sub.rn.f32 d0, $3, $1;
          sub.rn.f32 d1, $4, $2;
          fma.rn.f32 d0, d0, d0, $5;
          fma.rn.f32 $0, d1, d1, d0;
        }
        """,
        constraints="=f,f,f,f,f,f",
        args=[x0, x1, q0, q1, acc],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _scaled_fp32x16_e2m1_quant_squared_error(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    err = _fp32x2_e2m1_quant_squared_error(x0, x1)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x2, x3)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x4, x5)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x6, x7)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x8, x9)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x10, x11)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x12, x13)
    err = _fp32x2_e2m1_quant_squared_error_acc(err, x14, x15)

    return err


@triton.jit
def _fp32x16_e2m1_quant_squared_error(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
    scale,
):
    squared_error = _scaled_fp32x16_e2m1_quant_squared_error(
        v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale, v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale, v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale, v14 * inv_scale, v15 * inv_scale,
    )
    return squared_error * (scale * scale)


@triton.jit
def _max_abs_16(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
):
    m0 = tl.maximum(tl.abs(v0), tl.abs(v1))
    m1 = tl.maximum(tl.abs(v2), tl.abs(v3))
    m2 = tl.maximum(tl.abs(v4), tl.abs(v5))
    m3 = tl.maximum(tl.abs(v6), tl.abs(v7))
    m4 = tl.maximum(tl.abs(v8), tl.abs(v9))
    m5 = tl.maximum(tl.abs(v10), tl.abs(v11))
    m6 = tl.maximum(tl.abs(v12), tl.abs(v13))
    m7 = tl.maximum(tl.abs(v14), tl.abs(v15))

    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    m45 = tl.maximum(m4, m5)
    m67 = tl.maximum(m6, m7)
    return tl.maximum(tl.maximum(m01, m23), tl.maximum(m45, m67))


@triton.jit
def _load_normalized_16_cols(ptr, block_offsets, block_mask, global_scale_inv):
    base_elem = block_offsets * 16

    v0 = tl.load(ptr + base_elem + 0, mask=block_mask, other=0.0).to(tl.float32)
    v1 = tl.load(ptr + base_elem + 1, mask=block_mask, other=0.0).to(tl.float32)
    v2 = tl.load(ptr + base_elem + 2, mask=block_mask, other=0.0).to(tl.float32)
    v3 = tl.load(ptr + base_elem + 3, mask=block_mask, other=0.0).to(tl.float32)
    v4 = tl.load(ptr + base_elem + 4, mask=block_mask, other=0.0).to(tl.float32)
    v5 = tl.load(ptr + base_elem + 5, mask=block_mask, other=0.0).to(tl.float32)
    v6 = tl.load(ptr + base_elem + 6, mask=block_mask, other=0.0).to(tl.float32)
    v7 = tl.load(ptr + base_elem + 7, mask=block_mask, other=0.0).to(tl.float32)
    v8 = tl.load(ptr + base_elem + 8, mask=block_mask, other=0.0).to(tl.float32)
    v9 = tl.load(ptr + base_elem + 9, mask=block_mask, other=0.0).to(tl.float32)
    v10 = tl.load(ptr + base_elem + 10, mask=block_mask, other=0.0).to(tl.float32)
    v11 = tl.load(ptr + base_elem + 11, mask=block_mask, other=0.0).to(tl.float32)
    v12 = tl.load(ptr + base_elem + 12, mask=block_mask, other=0.0).to(tl.float32)
    v13 = tl.load(ptr + base_elem + 13, mask=block_mask, other=0.0).to(tl.float32)
    v14 = tl.load(ptr + base_elem + 14, mask=block_mask, other=0.0).to(tl.float32)
    v15 = tl.load(ptr + base_elem + 15, mask=block_mask, other=0.0).to(tl.float32)

    return (
        v0 * global_scale_inv, v1 * global_scale_inv,
        v2 * global_scale_inv, v3 * global_scale_inv,
        v4 * global_scale_inv, v5 * global_scale_inv,
        v6 * global_scale_inv, v7 * global_scale_inv,
        v8 * global_scale_inv, v9 * global_scale_inv,
        v10 * global_scale_inv, v11 * global_scale_inv,
        v12 * global_scale_inv, v13 * global_scale_inv,
        v14 * global_scale_inv, v15 * global_scale_inv,
    )


SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024}, num_warps=32),
]


@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=[
        "NUM_BLOCKS",
        "LOWER_BOUND",
        "NUM_CANDIDATES",
        "MAX_SCALE_RAW",
        "BLOCKS_PER_OUT",
        "BLOCKS_PER_OUT_PAD",
        "IS_SWIZZLE_SCALE",
    ],
)
@triton.jit
def _scalesweep_mse_nvfp4_quant_simulate_kernel(
    input_ptr,
    output_scale_ptr,
    output_i32_ptr,
    global_scale_inv_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS_PER_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MAX_SCALE_RAW: tl.constexpr,
    IS_SWIZZLE_SCALE: tl.constexpr,
    BLOCKS_PER_OUT_PAD: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)
    pid = tl.program_id(0)
    block_offsets = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_offsets < NUM_BLOCKS

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(input_ptr, block_offsets, block_mask, global_scale_inv)

    abs_max = _max_abs_16(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )
    base_scale = abs_max * (1.0 / 6.0)
    base_raw = base_scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)

    best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
    best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)

    for i in tl.static_range(0, NUM_CANDIDATES):
        raw_i = tl.minimum(
            tl.maximum(base_raw + LOWER_BOUND + i, 1),
            MAX_SCALE_RAW,
        ).to(tl.uint8)
        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = _fp32x16_e2m1_quant_squared_error(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            inv_scale_i,
            scale_i,
        )

        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    scale_offsets = block_offsets
    if IS_SWIZZLE_SCALE:
        scale_offsets = swizzled_scale_offsets(block_offsets, BLOCKS_PER_OUT, BLOCKS_PER_OUT_PAD)
    tl.store(output_scale_ptr + scale_offsets, best_scale_fp8, mask=block_mask)

    inv_scale = 1.0 / best_scale_fp8.to(tl.float32)
    lo, hi = _fp32x16_to_e2m1_u32x2(
        v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale,
        v4 * inv_scale, v5 * inv_scale, v6 * inv_scale, v7 * inv_scale,
        v8 * inv_scale, v9 * inv_scale, v10 * inv_scale, v11 * inv_scale,
        v12 * inv_scale, v13 * inv_scale, v14 * inv_scale, v15 * inv_scale,
    )

    output_i32_offsets = block_offsets * 2
    tl.store(output_i32_ptr + output_i32_offsets, lo, mask=block_mask)
    tl.store(output_i32_ptr + output_i32_offsets + 1, hi, mask=block_mask)


def _scalesweep_mse_nvfp4_quant_simulate(
    input: torch.Tensor,
    global_scale_inv: torch.Tensor,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input.shape
    num_blocks = input.numel() // BLOCK_SIZE
    blocks_per_out = n // BLOCK_SIZE

    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
    )
    output_i32 = output.view(torch.int32)

    grid = lambda meta: (triton.cdiv(num_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    _scalesweep_mse_nvfp4_quant_simulate_kernel[grid](
        input,
        output_scale,
        output_i32,
        global_scale_inv,
        num_blocks,
        BLOCKS_PER_OUT=blocks_per_out,
        LOWER_BOUND=LOWER_BOUND,
        NUM_CANDIDATES=UPPER_BOUND - LOWER_BOUND + 1,
        MAX_SCALE_RAW=REF_MAX_SCALE_RAW,
        IS_SWIZZLE_SCALE=is_sf_swizzled_layout,
        BLOCKS_PER_OUT_PAD=round_up(blocks_per_out, 4),
    )
    return output.view(torch.uint8), output_scale


def scalesweep_mse_nvfp4_quant_simulate(
    input: torch.Tensor,
    global_scale_inv: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    backend: str = "scalesweep_mse",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input to packed simulated NVFP4 using ScaleSweep MSE.

    ``global_scale_inv`` is the reciprocal global scale. For this ScaleSweep
    MSE variant the global scale is computed with FP8_MAX=256.
    """
    del backend

    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape

    assert n % BLOCK_SIZE == 0, (
        f"last dim has to be multiple of {BLOCK_SIZE}, but got {n}."
    )
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )
    assert global_scale_inv.dtype == torch.float32, (
        f"global_scale_inv.dtype needs to be fp32 but got {global_scale_inv.dtype}."
    )

    output, output_scale = _scalesweep_mse_nvfp4_quant_simulate(
        input.contiguous(),
        global_scale_inv,
        is_sf_swizzled_layout,
    )
    return output, output_scale


scalesweep_mse_nvfp4_quant = scalesweep_mse_nvfp4_quant_simulate
scaled_fp4_quant = scalesweep_mse_nvfp4_quant_simulate
scaled_fp4_quant_simulate = scalesweep_mse_nvfp4_quant_simulate
