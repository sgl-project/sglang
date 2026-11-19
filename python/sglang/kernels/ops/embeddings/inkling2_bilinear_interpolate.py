import triton
import triton.language as tl


@triton.jit
def _linear_indices_and_weights(
    output_index, target_length, SOURCE_LENGTH: tl.constexpr
):
    coordinate = (
        tl.div_rn(float(SOURCE_LENGTH) * (output_index + 0.5), target_length) - 0.5
    )
    coordinate = tl.maximum(0.0, tl.minimum(coordinate, SOURCE_LENGTH - 1.0))
    source0 = tl.floor(coordinate).to(tl.int64)
    source1 = tl.minimum(source0 + 1, SOURCE_LENGTH - 1)
    weight1 = coordinate - source0.to(tl.float32)
    weight0 = 1.0 - weight1
    return (source0, source1, weight0, weight1)


@triton.jit
def _bilinear_interpolate_row(
    input_ptr,
    output_y,
    output_x,
    target_height,
    target_width,
    dim,
    mask,
    SOURCE_HEIGHT: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    source_y0, source_y1, weight_y0, weight_y1 = _linear_indices_and_weights(
        output_y, target_height, SOURCE_HEIGHT
    )
    source_x0, source_x1, weight_x0, weight_x1 = _linear_indices_and_weights(
        output_x, target_width, SOURCE_WIDTH
    )
    offset00 = (source_y0 * SOURCE_WIDTH + source_x0) * EMBED_DIM + dim
    offset01 = (source_y0 * SOURCE_WIDTH + source_x1) * EMBED_DIM + dim
    offset10 = (source_y1 * SOURCE_WIDTH + source_x0) * EMBED_DIM + dim
    offset11 = (source_y1 * SOURCE_WIDTH + source_x1) * EMBED_DIM + dim
    value00 = tl.load(input_ptr + offset00, mask=mask, other=0.0).to(tl.float32)
    value01 = tl.load(input_ptr + offset01, mask=mask, other=0.0).to(tl.float32)
    value10 = tl.load(input_ptr + offset10, mask=mask, other=0.0).to(tl.float32)
    value11 = tl.load(input_ptr + offset11, mask=mask, other=0.0).to(tl.float32)
    top = tl.where(
        weight_x0 >= weight_x1,
        tl.fma(value00, weight_x0, value01 * weight_x1),
        tl.fma(value01, weight_x1, value00 * weight_x0),
    )
    bottom = tl.where(
        weight_x0 >= weight_x1,
        tl.fma(value10, weight_x0, value11 * weight_x1),
        tl.fma(value11, weight_x1, value10 * weight_x0),
    )
    return tl.where(
        weight_y0 >= weight_y1,
        tl.fma(top, weight_y0, bottom * weight_y1),
        tl.fma(bottom, weight_y1, top * weight_y0),
    )


@triton.jit(
    do_not_specialize=["num_rows"],
    do_not_specialize_on_alignment=[
        "output_y_ptr",
        "output_x_ptr",
        "target_heights_ptr",
        "target_widths_ptr",
    ],
)
def bilinear_interpolate_rows_fwd_kernel(
    input_ptr,
    output_y_ptr,
    output_x_ptr,
    target_heights_ptr,
    target_widths_ptr,
    output_ptr,
    num_rows: int,
    SOURCE_HEIGHT: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    EMBED_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dim = (tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)).to(tl.int64)
    row_mask = row < num_rows
    mask = row_mask & (dim < EMBED_DIM)
    output_y = tl.load(output_y_ptr + row, mask=row_mask, other=0).to(tl.float32)
    output_x = tl.load(output_x_ptr + row, mask=row_mask, other=0).to(tl.float32)
    target_height = tl.load(target_heights_ptr + row, mask=row_mask, other=1).to(
        tl.float32
    )
    target_width = tl.load(target_widths_ptr + row, mask=row_mask, other=1).to(
        tl.float32
    )
    value = _bilinear_interpolate_row(
        input_ptr,
        output_y,
        output_x,
        target_height,
        target_width,
        dim,
        mask,
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
        EMBED_DIM,
    )
    tl.store(output_ptr + row * EMBED_DIM + dim, value, mask=mask)


def bilinear_interpolate_rows(position_grid, y, x, heights, widths):
    rows, width = y.numel(), position_grid.shape[-1]
    output = position_grid.new_empty((rows, width))
    bilinear_interpolate_rows_fwd_kernel[(rows, 1)](
        position_grid,
        y,
        x,
        heights,
        widths,
        output,
        rows,
        SOURCE_HEIGHT=position_grid.shape[0],
        SOURCE_WIDTH=position_grid.shape[1],
        EMBED_DIM=width,
        BLOCK_DIM=triton.next_power_of_2(width),
        num_warps=1,
    )
    return output
