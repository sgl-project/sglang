"""Reconstruct the signed INT8 DSv4.1 page format in one Ascend graph launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_pages(
    Values,
    Scales,
    Selectors,
    Factors,
    Rope,
    Pages,
    Output,
    N,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    SCALES: tl.constexpr,
    ROPE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    SELECTOR_STRIDE: tl.constexpr,
    FACTOR_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    col = tl.arange(0, COLS)[None, :]
    for page in range(tl.program_id(0), N, tl.num_programs(0)):
        source = tl.load(Pages + page).to(tl.int32)
        # Only scratch page zero is read for padding. Other unreferenced
        # slots are mapped to zero in the block table and need no stores.
        if (page == 0) | (source != 0):
            for start in range(0, PAGE, ROWS):
                row = start + tl.arange(0, ROWS)[:, None]
                src = source.to(tl.int64) * PAGE + row
                dst = (page.to(tl.int64) * PAGE + row) * (WIDTH + ROPE)
                valid = (row < PAGE) & (col < WIDTH)
                codes = tl.load(Values + src * VALUE_STRIDE + col, valid, 0)
                pairs = tl.load(Selectors + src * SELECTOR_STRIDE + col // 2, valid, 0)
                selector = (pairs.to(tl.int32) >> ((col % 2) * 4)) & 15
                scale_col = (col // GROUP) * SCALES + selector
                bits = tl.load(Scales + src * SCALE_STRIDE + scale_col, valid, 0)
                magnitude = (bits.to(tl.int32) & 32767).to(tl.uint16)
                fp16 = magnitude.to(tl.float16, bitcast=True).to(tl.float32)
                bf16 = magnitude.to(tl.bfloat16, bitcast=True).to(tl.float32)
                scale = tl.where(bits < 0, bf16, fp16)
                factor = tl.load(Factors + src * FACTOR_STRIDE, row < PAGE, 0)
                value = (codes.to(tl.float32) * scale) * factor.to(tl.float32)
                limit = 3.3895313892515355e38
                value = tl.maximum(tl.minimum(value, limit), -limit)
                tl.store(Output + dst + col, value.to(tl.bfloat16), valid)
                if ROPE:
                    rc = tl.arange(0, ROPE)[None, :]
                    val = tl.load(Rope + src * ROPE_STRIDE + rc, row < PAGE, 0)
                    tl.store(Output + dst + WIDTH + rc, val, row < PAGE)


def decode_pages(values, metadata, pages, group_size, scales_per_group, rope=None):
    """Decode live physical pages and copy optional BF16 rotary channels verbatim."""
    page_size = values.shape[1]
    width = values.shape[-1]
    rope_width = 0 if rope is None else rope.shape[-1]
    output = values.new_empty(
        (pages.numel(), *values.shape[1:-1], width + rope_width), dtype=torch.bfloat16
    )
    if not pages.numel():
        return output
    scale_bytes = (width // group_size) * scales_per_group * 2
    scales = metadata[..., :scale_bytes].view(torch.int16)
    selectors = metadata[..., scale_bytes:-2]
    factors = metadata[..., -2:].view(torch.bfloat16)
    _decode_pages[(min(48, pages.numel()),)](
        values,
        scales,
        selectors,
        factors,
        rope if rope is not None else factors,
        pages,
        output,
        pages.numel(),
        page_size,
        width,
        group_size,
        scales_per_group,
        rope_width,
        values.stride(1),
        scales.stride(1),
        selectors.stride(1),
        factors.stride(1),
        0 if rope is None else rope.stride(1),
        4,
        triton.next_power_of_2(width),
    )
    return output
