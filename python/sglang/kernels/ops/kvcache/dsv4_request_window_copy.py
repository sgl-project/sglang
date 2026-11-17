"""Copy packed request-window KV rows without materializing byte index grids."""

import triton
import triton.language as tl


@triton.jit
def _row_offset(loc, col, PAGE, DATA, DATA_STRIDE, SCALE_BASE, SCALE_STRIDE):
    # Byte `col` of token `loc`'s data-then-scales, from its page start; padding is untouched.
    slot = loc % PAGE
    return tl.where(
        col < DATA,
        slot * DATA_STRIDE + col,
        SCALE_BASE + slot * SCALE_STRIDE + col - DATA,
    )


@triton.jit
def _copy_packed_rows(
    src,
    dst,
    src_loc,
    dst_loc,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    SRC_LOC_STRIDE: tl.constexpr,
    DST_LOC_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    DATA: tl.constexpr,
    SCALE: tl.constexpr,
    DATA_STRIDE: tl.constexpr,
    SCALE_BASE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(src_loc + row * SRC_LOC_STRIDE).to(tl.int64)
    target = tl.load(dst_loc + row * DST_LOC_STRIDE).to(tl.int64)
    col = tl.arange(0, BLOCK)
    source_offset = _row_offset(
        source, col, PAGE, DATA, DATA_STRIDE, SCALE_BASE, SCALE_STRIDE
    )
    target_offset = _row_offset(
        target, col, PAGE, DATA, DATA_STRIDE, SCALE_BASE, SCALE_STRIDE
    )
    value = tl.load(
        src + source // PAGE * SRC_STRIDE + source_offset,
        mask=col < DATA + SCALE,
        other=0,
    )
    tl.store(
        dst + target // PAGE * DST_STRIDE + target_offset,
        value,
        mask=col < DATA + SCALE,
    )


def copy_packed_rows(src, dst, src_loc, dst_loc, *, page_size, layout):
    _copy_packed_rows[(src_loc.numel(),)](
        src,
        dst,
        src_loc,
        dst_loc,
        src.stride(0),
        dst.stride(0),
        src_loc.stride(0),
        dst_loc.stride(0),
        page_size,
        layout.data_bytes,
        layout.scale_bytes,
        *layout.row_addressing(page_size),
        triton.next_power_of_2(layout.bytes_per_token),
        num_warps=4,
    )


@triton.jit
def _commit_packed_rows(
    src,
    dst,
    tags,
    src_loc,
    req,
    pos,
    keep,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    CAPACITY: tl.constexpr,
    DATA: tl.constexpr,
    SCALE: tl.constexpr,
    DATA_STRIDE: tl.constexpr,
    SCALE_BASE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Discarded rows neither read KV nor write the shared sink row.
    if tl.load(keep + row):
        source = tl.load(src_loc + row).to(tl.int64)
        position = tl.load(pos + row).to(tl.int64)
        target = tl.load(req + row).to(tl.int64) * CAPACITY + position % CAPACITY
        col = tl.arange(0, BLOCK)
        source_offset = _row_offset(
            source, col, PAGE, DATA, DATA_STRIDE, SCALE_BASE, SCALE_STRIDE
        )
        target_offset = _row_offset(
            target, col, PAGE, DATA, DATA_STRIDE, SCALE_BASE, SCALE_STRIDE
        )
        value = tl.load(
            src + source // PAGE * SRC_STRIDE + source_offset,
            mask=col < DATA + SCALE,
            other=0,
        )
        tl.store(
            dst + target // PAGE * DST_STRIDE + target_offset,
            value,
            mask=col < DATA + SCALE,
        )
        tl.store(tags + target, position)


def commit_packed_rows(src, dst, tags, window, *, page_size, capacity, layout):
    _commit_packed_rows[(window.write_loc.numel(),)](
        src,
        dst,
        tags,
        window.write_loc,
        window.req,
        window.pos,
        window.commit_mask,
        src.stride(0),
        dst.stride(0),
        page_size,
        capacity,
        layout.data_bytes,
        layout.scale_bytes,
        *layout.row_addressing(page_size),
        triton.next_power_of_2(layout.bytes_per_token),
        num_warps=4,
    )
