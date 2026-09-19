# SPDX-License-Identifier: Apache-2.0
"""Dynamic MXFP4 (1x32 block, E8M0 scale) activation quantization, in Triton.

Values are packed two E2M1 nibbles per byte: ``[M, N // 2]`` uint8 values and
``[M, ceil(N / 32)]`` uint8 scales. The scale tensor is allocated transposed,
so its row stride is 1 -- the MXFP4 GEMMs that consume it read a column of
block scales per row-tile, and callers passing straight to ``gemm_afp4wfp4``
depend on that layout.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl

# Fixed by the MXFP4 spec. Not a tuning knob.
MXFP4_QUANT_BLOCK_SIZE = 32


@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK)

    # Shared scale: round the block amax up to a power of two by adding half an
    # exponent step to the mantissa and truncating, then take its exponent. The
    # -2 puts the block's largest value at the top of the E2M1 range (max 6).
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    qx = x * tl.exp2(-scale_e8m0_unbiased)

    # E2M1: S000 +/-0, S001 +/-0.5 (denormal), S010..S111 = 1, 1.5, 2, 3, 4, 6.
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s  # work on the magnitude; the sign goes back on at the end

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormals: add a constant that lands the value's bits in the low mantissa,
    # so the subtraction below leaves exactly the E2M1 code.
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normals: rebias the exponent and round to nearest-even in one add pair.
    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Saturating values take 0x7 (+/-6), the largest E2M1 magnitude.
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    e2m1_value = e2m1_value | sign_lp.to(tl.uint8)

    # Pack adjacent pairs: even element in the low nibble, odd in the high one.
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, QUANT_BLOCK // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = (evens | (odds << 4)).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def _dynamic_mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_ITER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    EVEN_M_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_n = tl.program_id(1) * NUM_ITER

    # int64 strides: M*N can exceed int32 on a prefill activation.
    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m_in, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n_in, tl.int64)
    stride_bs_m = tl.cast(stride_bs_m_in, tl.int64)
    stride_bs_n = tl.cast(stride_bs_n_in, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    for pid_n in tl.range(start_n, start_n + NUM_ITER, num_stages=NUM_STAGES):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_offs = offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n

        if EVEN_M_N:
            x = tl.load(x_ptr + x_offs, cache_modifier=".cg").to(tl.float32)
        else:
            x_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
            x = tl.load(
                x_ptr + x_offs, mask=x_mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)

        out_tensor, bs_e8m0 = _mxfp4_quant_op(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, QUANT_BLOCK
        )

        out_offs_n = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )
        if EVEN_M_N:
            tl.store(x_fp4_ptr + out_offs, out_tensor)
        else:
            out_mask = (offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
            tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        if EVEN_M_N:
            tl.store(bs_ptr + bs_offs, bs_e8m0)
        else:
            n_blocks = (N + QUANT_BLOCK - 1) // QUANT_BLOCK
            bs_mask = (offs_m < M)[:, None] & (bs_offs_n < n_blocks)[None, :]
            tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


@functools.lru_cache(maxsize=256)
def _refill_tile(M: int, N: int, num_sms: int) -> Tuple[int, int]:
    # Widest tile first so each CTA keeps a full cache line per row, widest
    # BLOCK_SIZE_N among ties. (0, 0) means keep the ladder's pick.
    reach = []
    small = []
    for bm in (64, 32, 16, 8, 4, 2):
        if bm > triton.next_power_of_2(M):
            continue
        for bn in (256, 128, 64, 32):
            # bm * bn < 512 leaves a tile too small to amortize its own setup.
            if bn > triton.next_power_of_2(N) or bm * bn < 512:
                continue
            grid = triton.cdiv(M, bm) * triton.cdiv(N, bn)
            (reach if grid >= num_sms else small).append((bm * bn, bn, grid, bm))
    if reach:
        _, bn, _, bm = max(reach)
    elif small:
        _, bn, _, bm = max(small, key=lambda c: (c[2], c[0], c[1]))
    else:
        return 0, 0
    return bm, bn


@functools.lru_cache(maxsize=8)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _ladder_tile(M: int, N: int) -> Tuple[int, int, int, int, int]:
    # The published ladder, sized for prefill activations.
    if M <= 32:
        num_iter, block_m, block_n, num_warps, num_stages = (
            1,
            triton.next_power_of_2(M),
            32,
            1,
            1,
        )
    elif N <= 16384:
        num_iter, block_m, block_n, num_warps, num_stages = 4, 32, 128, 4, 2
    else:
        num_iter, block_m, block_n, num_warps, num_stages = 4, 64, 64, 4, 2

    if N <= 1024:
        num_iter, num_stages, num_warps = 1, 1, 4
        # BLOCK_SIZE_N has to stay a multiple of the 32-wide quant block.
        block_n = max(32, min(256, triton.next_power_of_2(N)))
        block_m = min(8, triton.next_power_of_2(M))

    return block_m, block_n, num_iter, num_stages, num_warps


def _select_tile(M: int, N: int, num_sms: int) -> Tuple[int, int, int, int, int]:
    block_m, block_n, num_iter, num_stages, num_warps = _ladder_tile(M, N)
    blocks = triton.cdiv(M, block_m) * triton.cdiv(N, block_n * num_iter)
    if blocks >= num_sms:
        return block_m, block_n, num_iter, num_stages, num_warps

    bm, bn = _refill_tile(M, N, num_sms)
    # The 512-element tile floor can leave every wider-grid candidate out, in
    # which case the re-pick covers less of the part than the ladder already
    # did (M = 6, N = 6144: 96 blocks against 192). Only switch on improvement.
    if bm and triton.cdiv(M, bm) * triton.cdiv(N, bn) > blocks:
        return bm, bn, 1, 1, 4
    return block_m, block_n, num_iter, num_stages, num_warps


def dynamic_mxfp4_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D activation to ``(values [M, N // 2], e8m0 scales [M, N / 32])``."""
    assert x.dim() == 2, f"expected a 2-D activation, got shape {tuple(x.shape)}"
    M, N = x.shape
    assert (N // 2) % 2 == 0, f"N // 2 must be even for nibble packing, got N={N}"

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_ITER, NUM_STAGES, NUM_WARPS = _select_tile(
        M, N, _num_sms(x.device.index or 0)
    )
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER))

    even_m_n = (
        M % BLOCK_SIZE_M == 0
        and N % (BLOCK_SIZE_N * NUM_ITER) == 0
        and N % MXFP4_QUANT_BLOCK_SIZE == 0
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_ITER=NUM_ITER,
        NUM_STAGES=NUM_STAGES,
        QUANT_BLOCK=MXFP4_QUANT_BLOCK_SIZE,
        EVEN_M_N=even_m_n,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return x_fp4, blockscale_e8m0
