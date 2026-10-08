# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""VAE-aligned tile grids and Hann blending for Kandinsky 6 video SR."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
from torch.nn import functional

VAE_SPATIAL_FACTOR: int = 16
VAE_TEMPORAL_FACTOR: int = 4

# trained base resolutions: visual_size -> [(H, W), ...]
RESOLUTIONS: dict[int, list[tuple[int, int]]] = {
    512: [(512, 512), (512, 768), (768, 512)],
}


class TileGrid(NamedTuple):
    """Tile starts are authoritative; gaps can differ near the frame edges."""

    tile_h: int
    tile_w: int
    tops: tuple[int, ...]
    lefts: tuple[int, ...]

    @property
    def n_h(self) -> int:
        """Number of tile rows."""
        return len(self.tops)

    @property
    def n_w(self) -> int:
        """Number of tile columns."""
        return len(self.lefts)

    @property
    def total_tiles(self) -> int:
        """Total number of tiles (``n_h * n_w``)."""
        return self.n_h * self.n_w


def extract_all_tiles(
    video: torch.Tensor,
    grid: TileGrid,
) -> list[torch.Tensor]:
    """Extract [T, C, tile_h, tile_w] views in row-major order."""
    tiles: list[torch.Tensor] = []
    for top in grid.tops:
        for left in grid.lefts:
            tile = video[:, :, top : top + grid.tile_h, left : left + grid.tile_w]
            tiles.append(tile)
    return tiles


def stitch_tiles_hanning(
    tiles: list[torch.Tensor],
    grid: TileGrid,
    original_h: int,
    original_w: int,
    scale: int = 1,
) -> torch.Tensor:
    """Blend row-major [C, T, H, W] tiles into a scale-enlarged frame."""
    first = tiles[0]
    c, t = first.shape[0], first.shape[1]
    hr_tile_h, hr_tile_w = first.shape[2], first.shape[3]
    device = first.device

    # nonzero endpoints keep the outermost tile pixels covered
    wy = torch.hann_window(hr_tile_h + 2, device=device)[1:-1]
    wx = torch.hann_window(hr_tile_w + 2, device=device)[1:-1]
    window = (wy[:, None] * wx[None, :])[None, None]
    del wy, wx

    out_h = original_h * scale
    out_w = original_w * scale
    pred_acc = torch.zeros(c, t, out_h, out_w, device=device)
    weight_acc = torch.zeros(1, 1, out_h, out_w, device=device)

    tile_idx = 0
    for top in grid.tops:
        for left in grid.lefts:
            y = top * scale
            x = left * scale
            tile = tiles[tile_idx]
            pred_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += tile * window
            weight_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += window
            tile_idx += 1

    # preserve tiny positive corner weights; only uncovered pixels stay zero
    covered = weight_acc > 0
    safe_weight = torch.where(covered, weight_acc, torch.ones_like(weight_acc))
    return torch.where(covered, pred_acc / safe_weight, torch.zeros_like(pred_acc))


def axis_positions_even(
    length: int, tile: int, min_overlap: float, snap: int
) -> tuple[int, ...]:
    """Cover the axis with the fewest evenly spaced, snap-aligned tiles.

    Fall back to pixel alignment for unaligned sizes; latent-grid conversion
    validates VAE alignment separately."""
    if tile <= 0 or snap <= 0 or not 0 <= min_overlap < 1:
        msg = f"invalid axis spec: tile={tile}, snap={snap}, min_overlap={min_overlap}"
        raise ValueError(msg)
    if tile >= length:
        return (0,)
    unit = snap if length % snap == 0 and tile % snap == 0 else 1
    span_units = (length - tile) // unit
    max_stride_units = max(1, math.floor(tile * (1.0 - min_overlap) / unit))
    count = math.ceil(span_units / max_stride_units) + 1
    return tuple(round(i * span_units / (count - 1)) * unit for i in range(count))


def compute_tile_grid_even(
    h: int,
    w: int,
    tile_hw: tuple[int, int],
    min_overlap: float,
    snap: int,
) -> TileGrid:
    """Build a grid with evenly distributed, aligned tile starts."""
    tile_h, tile_w = tile_hw
    return TileGrid(
        tile_h=tile_h,
        tile_w=tile_w,
        tops=axis_positions_even(h, tile_h, min_overlap, snap),
        lefts=axis_positions_even(w, tile_w, min_overlap, snap),
    )


def latent_tile_grid_from_pixel_grid(
    pixel_grid: TileGrid, spatial_factor: int
) -> TileGrid:
    """Convert an aligned pixel grid to the corresponding latent grid."""
    values = (pixel_grid.tile_h, pixel_grid.tile_w, *pixel_grid.tops, *pixel_grid.lefts)
    if any(value % spatial_factor for value in values):
        raise ValueError(
            f"Pixel tile grid (tile {pixel_grid.tile_h}x{pixel_grid.tile_w}, "
            f"tops={pixel_grid.tops}, lefts={pixel_grid.lefts}) not aligned to VAE spatial factor "
            f"{spatial_factor}."
        )
    return TileGrid(
        pixel_grid.tile_h // spatial_factor,
        pixel_grid.tile_w // spatial_factor,
        tuple(top // spatial_factor for top in pixel_grid.tops),
        tuple(left // spatial_factor for left in pixel_grid.lefts),
    )


def closest_base_resolution(
    h: int,
    w: int,
    visual_size: int,
    resolutions: Mapping[int, Sequence[tuple[int, int]]] = RESOLUTIONS,
) -> tuple[int, int]:
    """Choose the trained base resolution with the closest aspect ratio."""
    if visual_size not in resolutions:
        raise ValueError(
            f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(resolutions)}"
        )
    ratio = w / h if h else 1.0
    return min(resolutions[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def resolve_scale_request(scale: float) -> tuple[int, float]:
    """Decompose total scale into (integer tiling scale, pixel pre-upscale)."""
    requested = float(scale)
    if requested == 2.25:  # the one supported fractional total
        return 2, 1.125
    if requested in (2.0, 4.0):
        return int(requested), 1.0
    raise ValueError("SR supports total scales 2, 4, and 2.25")


def pre_upscale_video(
    video: torch.Tensor, factor: float, spatial_multiple: int
) -> torch.Tensor:
    """Bilinearly enlarge [T, C, H, W] uint8 video, aligning target dimensions."""
    if video.ndim != 4:
        raise ValueError(f"video must have rank 4 [T,C,H,W], got {tuple(video.shape)}")
    if factor <= 0 or spatial_multiple <= 0:
        raise ValueError("factor and spatial_multiple must be positive")
    height, width = video.shape[-2:]
    target_h = max(
        spatial_multiple, round(height * factor / spatial_multiple) * spatial_multiple
    )
    target_w = max(
        spatial_multiple, round(width * factor / spatial_multiple) * spatial_multiple
    )
    resized = functional.interpolate(
        video.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.round_().clamp_(0, 255).to(torch.uint8)


def pad_to_spatial_factor(
    video: torch.Tensor, spatial_factor: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad bottom/right to the VAE spatial factor; return the original size.

    Crop the scaled padding after SR to preserve the source framing."""
    if video.ndim != 4:
        raise ValueError(f"video must have rank 4 [T,C,H,W], got {tuple(video.shape)}")
    height, width = video.shape[-2:]
    pad_h = (-height) % spatial_factor
    pad_w = (-width) % spatial_factor
    if pad_h == 0 and pad_w == 0:
        return video, (height, width)
    padded = functional.pad(video, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)
