"""NVFP4 KV-cache quantization with a per-block 4-over-6 scale search.

Standard NVFP4 maps each 16-value block's max to the largest E2M1 value (6).
The 4-over-6 search also tries mapping the max to 4, which gives a finer grid
near the top of the block, and keeps whichever candidate reconstructs the
block with lower squared error. The output layout matches FlashInfer's
``nvfp4_kv_quantize``, so the cache layout and attention kernels are unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NVFP4_BLOCK_SIZE = 16


@triton.jit
def _round_to_e2m1(ax):
    """Round |x| to the nearest E2M1 magnitude, ties to even (cvt.rn.satfinite)."""
    q = tl.where(ax <= 0.25, 0.0, 0.5)
    q = tl.where(ax >= 0.75, 1.0, q)
    q = tl.where(ax > 1.25, 1.5, q)
    q = tl.where(ax >= 1.75, 2.0, q)
    q = tl.where(ax > 2.5, 3.0, q)
    q = tl.where(ax >= 3.5, 4.0, q)
    q = tl.where(ax > 5.0, 6.0, q)
    return q


@triton.jit
def _e2m1_code(q_abs):
    """Map an E2M1 magnitude (0, 0.5, 1, 1.5, 2, 3, 4, 6) to its 3-bit code."""
    code = tl.where(q_abs >= 0.5, 1, 0)
    code = tl.where(q_abs >= 1.0, 2, code)
    code = tl.where(q_abs >= 1.5, 3, code)
    code = tl.where(q_abs >= 2.0, 4, code)
    code = tl.where(q_abs >= 3.0, 5, code)
    code = tl.where(q_abs >= 4.0, 6, code)
    code = tl.where(q_abs >= 6.0, 7, code)
    return code


@triton.jit
def _candidate(x, amax, inv_global, denominator):
    """Return (fp8 block scale, quant multiplier, squared error) for one candidate."""
    sf = (amax / denominator * inv_global).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    sf_f = sf.to(tl.float32)
    nonzero = sf_f != 0.0
    mult = tl.where(nonzero, inv_global / tl.where(nonzero, sf_f, 1.0), 0.0)
    q = _round_to_e2m1(tl.abs(x) * mult[:, None])
    recon = q * (sf_f / inv_global)[:, None]
    diff = recon - tl.abs(x)
    err = tl.sum(diff * diff, axis=1)
    return sf, mult, err


@triton.jit
def _nvfp4_kv_quantize_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    global_scale_ptr,
    num_blocks,
    FOUR_OVER_SIX: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    mask = blk < num_blocks
    cols = tl.arange(0, 16)
    x = tl.load(
        x_ptr + blk[:, None] * 16 + cols[None, :], mask=mask[:, None], other=0.0
    ).to(tl.float32)
    # The cache convention is x = x_fp4 * block_scale * global_scale.
    inv_global = 1.0 / tl.load(global_scale_ptr)
    amax = tl.max(tl.abs(x), axis=1)

    sf, mult, err = _candidate(x, amax, inv_global, 6.0)
    if FOUR_OVER_SIX:
        sf4, mult4, err4 = _candidate(x, amax, inv_global, 4.0)
        use4 = err4 < err
        sf = tl.where(use4, sf4, sf)
        mult = tl.where(use4, mult4, mult)

    q_abs = _round_to_e2m1(tl.abs(x) * mult[:, None])
    code = _e2m1_code(q_abs) | tl.where(x < 0.0, 8, 0)
    lo, hi = tl.split(tl.reshape(code, (BLOCKS_PER_PROGRAM, 8, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    tl.store(
        out_ptr + blk[:, None] * 8 + tl.arange(0, 8)[None, :],
        packed,
        mask=mask[:, None],
    )
    tl.store(sf_ptr + blk, sf.to(tl.uint8, bitcast=True), mask=mask)


def nvfp4_kv_quantize_triton(
    x: torch.Tensor, global_scale: torch.Tensor, four_over_six: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` of shape [R, N] to packed NVFP4 with linear block scales.

    Args:
        x: BF16 or FP16 tensor of shape [R, N], N divisible by 16.
        global_scale: FP32 tensor with one element, on ``x``'s device.
        four_over_six: Choose between max/6 and max/4 block scales per block
            by squared reconstruction error. False gives the standard max/6.

    Returns:
        (packed FP4 data [R, N // 2] uint8, FP8 E4M3 block scales
        [R, N // 16] as uint8), the layout of FlashInfer's nvfp4_kv_quantize.
    """
    assert x.dim() == 2 and x.shape[1] % NVFP4_BLOCK_SIZE == 0, x.shape
    x = x.contiguous()
    rows, cols = x.shape
    out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(
        (rows, cols // NVFP4_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )
    num_blocks = rows * cols // NVFP4_BLOCK_SIZE
    if num_blocks == 0:
        return out, sf
    blocks_per_program = 64
    grid = (triton.cdiv(num_blocks, blocks_per_program),)
    _nvfp4_kv_quantize_kernel[grid](
        x,
        out,
        sf,
        global_scale.reshape(-1).to(torch.float32),
        num_blocks,
        FOUR_OVER_SIX=four_over_six,
        BLOCKS_PER_PROGRAM=blocks_per_program,
    )
    return out, sf
