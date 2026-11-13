import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def _as_float32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


_SUPPORTED_HEAD_DIMS = (4, 8, 16, 32, 64, 128)

_INV_FREQS = tl.constexpr(
    tuple(
        (
            _as_float32(10000.0 ** (-dim / supported_head_dim))
            for supported_head_dim in _SUPPORTED_HEAD_DIMS
            for dim in range(0, supported_head_dim, 4)
        )
    )
)


@triton.jit
def _rope_cos_sin(positions_ptr, token_index, pair, HEAD_DIM: tl.constexpr):
    if HEAD_DIM == 4:
        inv_freq_offset: tl.constexpr = 0
    elif HEAD_DIM == 8:
        inv_freq_offset: tl.constexpr = 1
    elif HEAD_DIM == 16:
        inv_freq_offset: tl.constexpr = 3
    elif HEAD_DIM == 32:
        inv_freq_offset: tl.constexpr = 7
    elif HEAD_DIM == 64:
        inv_freq_offset: tl.constexpr = 15
    elif HEAD_DIM == 128:
        inv_freq_offset: tl.constexpr = 31
    else:
        tl.static_assert(False, "unsupported head dimension")
    freq_index = pair % (HEAD_DIM // 4)
    inv_freq = tl.zeros(freq_index.shape, tl.float32)
    for index in tl.static_range(0, HEAD_DIM // 4):
        inv_freq = tl.where(
            freq_index == index, _INV_FREQS[inv_freq_offset + index], inv_freq
        )
    axis_offset = tl.where(pair < HEAD_DIM // 4, 1, 0)
    position = tl.load(
        positions_ptr + 2 * token_index + axis_offset.to(tl.int64),
        mask=pair < HEAD_DIM // 2,
        other=0,
    ).to(tl.float32)
    angle = position * inv_freq
    cos = libdevice.cos(angle)
    sin = libdevice.sin(angle)
    return (cos, sin)


@triton.jit
def _rotate_pair(even, odd, cos, sin):
    return (tl.fma(even, cos, -odd * sin), tl.fma(odd, cos, even * sin))


@triton.jit(
    do_not_specialize=["num_tokens"], do_not_specialize_on_alignment=["positions_ptr"]
)
def apply_rotary_qk_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    out_q_ptr,
    out_k_ptr,
    num_tokens: int,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= num_tokens:
        return
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)[:, None]
    pair = tl.arange(0, BLOCK_PAIRS)[None, :]
    mask = (head < NUM_HEADS) & (pair < HEAD_DIM // 2)
    token_index = token.to(tl.int64)
    head_index = head.to(tl.int64)
    pair_index = pair.to(tl.int64)
    pair_offset = (token_index * NUM_HEADS + head_index) * HEAD_DIM + 2 * pair_index
    cos, sin = _rope_cos_sin(positions_ptr, token_index, pair, HEAD_DIM)
    q_even = tl.load(q_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)
    q_odd = tl.load(q_ptr + pair_offset + 1, mask=mask, other=0.0).to(tl.float32)
    k_even = tl.load(k_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_ptr + pair_offset + 1, mask=mask, other=0.0).to(tl.float32)
    out_q_even, out_q_odd = _rotate_pair(q_even, q_odd, cos, sin)
    out_k_even, out_k_odd = _rotate_pair(k_even, k_odd, cos, sin)
    tl.store(out_q_ptr + pair_offset, out_q_even, mask=mask)
    tl.store(out_q_ptr + pair_offset + 1, out_q_odd, mask=mask)
    tl.store(out_k_ptr + pair_offset, out_k_even, mask=mask)
    tl.store(out_k_ptr + pair_offset + 1, out_k_odd, mask=mask)


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


def apply_rotary_qk(q, k, positions):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    num_tokens, num_heads, head_dim = q.shape
    block_heads = min(4, triton.next_power_of_2(num_heads))
    apply_rotary_qk_kernel[(num_tokens, triton.cdiv(num_heads, block_heads))](
        q,
        k,
        positions,
        out_q,
        out_k,
        num_tokens,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=triton.next_power_of_2(head_dim // 2),
        num_warps=4,
    )
    return out_q, out_k


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
