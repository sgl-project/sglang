# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 VAE tile assembly in one pass, bit-identical to the eager chain.

The released decoder blends each tile with the one above, then with the one to
its left, crops it and copies it into the canvas: about eight small launches a
tile. Per output pixel that chain is

    v   = tile            or  fl(fl(up * a_y) + fl(tile * b_y))    top overlap
    out = v               or  fl(fl(left * a_x) + fl(v * b_x))     left overlap

with ``up`` / ``left`` the neighbours as decoded (not their blended versions)
and the ramp weights of ``blend``. Products and sums stay separate FP32
operations (``enable_fp_fusion=False``), as aten rounds them.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_BLOCK = 1024


@triton.jit
def _tile_assemble_kernel(
    out_ptr,
    tiles_ptr,
    y_index_ptr,
    x_index_ptr,
    y_weight_ptr,
    x_weight_ptr,
    planes,
    tile_h,
    tile_w,
    out_h,
    out_w,
    j_max,
    BLOCK: tl.constexpr,
):
    plane = tl.program_id(0).to(tl.int64)
    pixel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = pixel < out_h * out_w
    y = pixel // out_w
    x = pixel % out_w
    # per canvas row / column: tile index, coordinate in it, coordinate in the
    # neighbour it blends with (-1: none)
    tile_i = tl.load(y_index_ptr + y * 3, mask=live, other=0)
    tile_y = tl.load(y_index_ptr + y * 3 + 1, mask=live, other=0)
    up_y = tl.load(y_index_ptr + y * 3 + 2, mask=live, other=-1)
    tile_j = tl.load(x_index_ptr + x * 3, mask=live, other=0)
    tile_x = tl.load(x_index_ptr + x * 3 + 1, mask=live, other=0)
    left_x = tl.load(x_index_ptr + x * 3 + 2, mask=live, other=-1)
    plane_size = tile_h * tile_w
    tile = (tile_i * j_max + tile_j).to(tl.int64)
    base = tiles_ptr + (tile * planes + plane) * plane_size
    value = tl.load(base + tile_y * tile_w + tile_x, mask=live, other=0.0)

    blend_y = live & (up_y >= 0)
    up = tl.load(
        tiles_ptr
        + ((tile - j_max) * planes + plane) * plane_size
        + up_y * tile_w
        + tile_x,
        mask=blend_y,
        other=0.0,
    )
    weight_a = tl.load(y_weight_ptr + y * 2, mask=blend_y, other=0.0)
    weight_b = tl.load(y_weight_ptr + y * 2 + 1, mask=blend_y, other=0.0)
    value = tl.where(blend_y, up * weight_a + value * weight_b, value)

    blend_x = live & (left_x >= 0)
    left = tl.load(
        tiles_ptr
        + ((tile - 1) * planes + plane) * plane_size
        + tile_y * tile_w
        + left_x,
        mask=blend_x,
        other=0.0,
    )
    weight_a = tl.load(x_weight_ptr + x * 2, mask=blend_x, other=0.0)
    weight_b = tl.load(x_weight_ptr + x * 2 + 1, mask=blend_x, other=0.0)
    value = tl.where(blend_x, left * weight_a + value * weight_b, value)
    tl.store(out_ptr + plane * out_h * out_w + pixel, value, mask=live)


def _axis_tables(
    size: int, overlaps: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[canvas, 3]`` int32 (tile, coordinate, neighbour coordinate) and
    ``[canvas, 2]`` FP32 (neighbour weight, own weight) along one axis."""
    index, weight = [], []
    last = len(overlaps)
    for tile, kept in enumerate(
        size - (overlaps[t] if t < last else 0) for t in range(last + 1)
    ):
        extent = min(size, overlaps[tile - 1]) if tile else 0
        coordinate = torch.arange(kept, device=device, dtype=torch.int32)
        neighbour = torch.where(coordinate < extent, size - extent + coordinate, -1)
        index.append(
            torch.stack(
                (torch.full_like(coordinate, tile), coordinate, neighbour), 1
            ).to(torch.int32)
        )
        ramp = torch.zeros(kept, 2, device=device, dtype=torch.float32)
        if extent:
            # the expressions of AutoencoderKL.blend, on the tiles' device; an
            # overlap wider than the part of this tile that is kept blends only
            # the kept rows
            positions = torch.arange(extent, device=device, dtype=torch.float32)
            blended = min(extent, kept)
            ramp[:blended, 0] = (1 - positions / extent)[:blended]
            ramp[:blended, 1] = (positions / extent)[:blended]
        weight.append(ramp)
    return torch.cat(index).contiguous(), torch.cat(weight).contiguous()


_TABLES: dict[tuple, tuple[torch.Tensor, ...]] = {}


def h3_vae_assemble_tiles(
    tiles: torch.Tensor,
    i_max: int,
    j_max: int,
    y_overlap: tuple[int, ...],
    x_overlap: tuple[int, ...],
) -> torch.Tensor:
    """Blend, crop and place ``[i_max * j_max, ..., h, w]`` FP32 tiles (row
    major) into the ``[..., H, W]`` canvas."""
    if not (
        tiles.is_cuda
        and tiles.dtype == torch.float32
        and tiles.is_contiguous()
        and tiles.shape[0] == i_max * j_max
        and len(y_overlap) == i_max - 1
        and len(x_overlap) == j_max - 1
    ):
        raise RuntimeError("unsupported input for the H3 VAE tile assembly")
    tile_h, tile_w = tiles.shape[-2:]
    key = (tile_h, tile_w, y_overlap, x_overlap, tiles.device)
    tables = _TABLES.get(key)
    if tables is None:
        tables = (
            *_axis_tables(tile_h, y_overlap, tiles.device),
            *_axis_tables(tile_w, x_overlap, tiles.device),
        )
        _TABLES[key] = tables
    y_index, y_weight, x_index, x_weight = tables
    out_h, out_w = y_index.shape[0], x_index.shape[0]
    lead = tiles.shape[1:-2]
    planes = 1
    for dim in lead:
        planes *= dim
    out = tiles.new_empty((*lead, out_h, out_w))
    with torch.cuda.device(tiles.device):
        _tile_assemble_kernel[(planes, triton.cdiv(out_h * out_w, _BLOCK))](
            out,
            tiles,
            y_index,
            x_index,
            y_weight,
            x_weight,
            planes,
            tile_h,
            tile_w,
            out_h,
            out_w,
            j_max,
            BLOCK=_BLOCK,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["h3_vae_assemble_tiles"]
