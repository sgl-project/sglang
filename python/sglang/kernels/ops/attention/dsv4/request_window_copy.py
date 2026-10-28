"""Copy packed request-window KV rows without materializing byte index grids."""

import triton
import triton.language as tl


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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(src_loc + row * SRC_LOC_STRIDE).to(tl.int64)
    target = tl.load(dst_loc + row * DST_LOC_STRIDE).to(tl.int64)
    col = tl.arange(0, BLOCK)
    # Each page contains all data rows, followed by all scale rows. Its
    # physical stride may also contain alignment padding, which is untouched.
    source_offset = tl.where(
        col < DATA,
        (source % PAGE) * DATA + col,
        PAGE * DATA + (source % PAGE) * SCALE + col - DATA,
    )
    target_offset = tl.where(
        col < DATA,
        (target % PAGE) * DATA + col,
        PAGE * DATA + (target % PAGE) * SCALE + col - DATA,
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
        triton.next_power_of_2(layout.bytes_per_token),
        num_warps=4,
    )
