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
    Lengths,
    Remapped,
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
    SCALE_COLS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    LEN_STRIDE: tl.constexpr,
    MAP_TABLE: tl.constexpr,
):
    cols = tl.arange(0, COLS)[None, :]
    byte_cols = tl.arange(0, COLS // 2)[None, :]
    palette_size: tl.constexpr = (WIDTH // GROUP) * SCALES
    scale_cols = tl.arange(0, SCALE_COLS)[None, :]
    pieces: tl.constexpr = triton.cdiv(PAGE, ROWS)
    for task in range(tl.program_id(0), N * pieces, tl.num_programs(0)):
        page = task // pieces
        if MAP_TABLE:
            entry = tl.maximum(page - 1, 0)
            owner, slot = entry // TABLE_WIDTH, entry % TABLE_WIDTH
            source_id = tl.load(
                Pages + owner.to(tl.int64) * TABLE_ROW + slot * TABLE_COL,
                page != 0,
                0,
            ).to(tl.int32)
            length = tl.load(Lengths + owner * LEN_STRIDE, page != 0, 0)
            live = (page != 0) & (slot * PAGE < length) & (source_id > 0)
            source = tl.where(live, source_id, 0)
            if (page != 0) & (task % pieces == 0):
                tl.store(
                    Remapped + entry, tl.where(live, page, tl.minimum(source_id, 0))
                )
        else:
            source = tl.load(Pages + page).to(tl.int32)
        if (page == 0) | (source != 0):
            row = (task % pieces) * ROWS + tl.arange(0, ROWS)[:, None]
            src = source.to(tl.int64) * PAGE + row
            dst = (page.to(tl.int64) * PAGE + row) * (WIDTH + ROPE)
            valid = (row < PAGE) & (cols < WIDTH)
            codes = tl.load(Values + src * VALUE_STRIDE + cols, valid, 0)
            # Read each scale and selector byte contiguously from device memory.
            # The per-channel palette lookup then stays within the vector tile.
            packed = tl.load(
                Selectors + src * SELECTOR_STRIDE + byte_cols,
                (row < PAGE) & (byte_cols < WIDTH // 2),
                0,
            ).to(tl.int32)
            selectors = tl.reshape(tl.join(packed & 15, packed >> 4), (ROWS, COLS))
            scale_index = tl.where(
                cols < WIDTH, (cols // GROUP) * SCALES + selectors, 0
            )
            bits = tl.load(
                Scales + src * SCALE_STRIDE + scale_cols,
                (row < PAGE) & (scale_cols < palette_size),
                0,
            )
            magnitude = (bits.to(tl.int32) & 32767).to(tl.uint16)
            half = magnitude.to(tl.float16, bitcast=True).to(tl.float32)
            bfloat = magnitude.to(tl.bfloat16, bitcast=True).to(tl.float32)
            palette = tl.where(bits < 0, bfloat, half)
            scale = tl.gather(palette, scale_index, axis=1)
            factor = tl.load(Factors + src * FACTOR_STRIDE, row < PAGE, 0)
            value = codes.to(tl.float32) * scale * factor.to(tl.float32)
            limit = 3.3895313892515355e38
            value = tl.maximum(tl.minimum(value, limit), -limit)
            tl.store(Output + dst + cols, value.to(tl.bfloat16), valid)
            if ROPE:
                rc = tl.arange(0, ROPE)[None, :]
                rope = tl.load(Rope + src * ROPE_STRIDE + rc, row < PAGE, 0)
                tl.store(Output + dst + WIDTH + rc, rope, row < PAGE)


def decode_pages(
    values,
    metadata,
    pages,
    group_size,
    scales_per_group,
    rope=None,
    *,
    lengths=None,
    remapped=None,
):
    """Decode live physical pages and copy optional BF16 rotary channels verbatim."""
    map_table = lengths is not None
    count = pages.numel() + int(map_table)
    page_size = values.shape[1]
    width = values.shape[-1]
    rope_width = 0 if rope is None else rope.shape[-1]
    output = values.new_empty(
        (count, *values.shape[1:-1], width + rope_width), dtype=torch.bfloat16
    )
    if not count:
        return output
    scale_bytes = (width // group_size) * scales_per_group * 2
    scales = metadata[..., :scale_bytes].view(torch.int16)
    selectors = metadata[..., scale_bytes:-2]
    factors = metadata[..., -2:].view(torch.bfloat16)
    columns = triton.next_power_of_2(width)
    # Bound the local gather tile even when the caller omits rotary channels.
    tile_rows = min(16, max(1, 4096 // columns), triton.next_power_of_2(page_size))
    _decode_pages[(min(48, count * triton.cdiv(page_size, tile_rows)),)](
        values,
        scales,
        selectors,
        factors,
        rope if rope is not None else factors,
        pages,
        lengths if map_table else pages,
        remapped if map_table else pages,
        output,
        count,
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
        tile_rows,
        columns,
        triton.next_power_of_2((width // group_size) * scales_per_group),
        max(pages.shape[1], 1) if map_table else 1,
        pages.stride(0) if map_table else 1,
        pages.stride(1) if map_table else 1,
        lengths.stride(0) if map_table else 1,
        map_table,
    )
    return output


def decode_logical_pages(
    values, metadata, table, lengths, group_size, scales_per_group, rope=None
):
    """Replay the logical-to-scratch mapping in the same launch as page reconstruction."""
    remapped = torch.empty(table.shape, dtype=torch.int32, device=table.device)
    output = decode_pages(
        values,
        metadata,
        table,
        group_size,
        scales_per_group,
        rope,
        lengths=lengths.reshape(-1),
        remapped=remapped,
    )
    return output, remapped
