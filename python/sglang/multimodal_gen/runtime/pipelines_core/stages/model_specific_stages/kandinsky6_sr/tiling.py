# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""Pure-torch tiling, tile-grid and scale helpers for tiled Kandinsky 6 video SR.

Ported from ``core/algo/tiling_utils.py``, ``pipeline/tile_grid.py``,
``pipeline/upscale_utils.py`` and the tile helpers of ``pipeline/sr_pipeline.py``
(``_closest_base_resolution``, ``_tile_geometry``,
``latent_tile_grid_from_pixel_grid``, ``_upsample_tiles_to_base``).

Unlike the reference there is no process-wide mutable state: the trained base
resolutions (:data:`RESOLUTIONS`) and the VAE compression factors
(:data:`VAE_SPATIAL_FACTOR`, :data:`VAE_TEMPORAL_FACTOR`) are plain module
constants, and every function that needs them takes them as (defaulted)
arguments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import torch
from torch.nn import functional

# --- Constants (formerly ``constants.py`` + ``constants.set_vae_factors`` global state) ---

#: VAE spatial downsample factor of the KVAE (``video-kvae``).
VAE_SPATIAL_FACTOR: int = 16
#: VAE temporal downsample factor of the KVAE (``video-kvae``).
VAE_TEMPORAL_FACTOR: int = 4

#: Trained base resolutions ``visual_size -> [(H, W), ...]``.
RESOLUTIONS: dict[int, list[tuple[int, int]]] = {
    512: [(512, 512), (512, 768), (768, 512)],
}

TileGridMode = Literal["legacy", "even"]


class TileGrid(NamedTuple):
    """Spatial tile grid parameters.

    ``tops`` and ``lefts`` are the authoritative per-axis tile start positions.
    They cover ``[0, length - tile]`` with the nominal stride; the **last**
    position is clamped to ``length - tile`` if a uniform stride would
    overshoot. This means consecutive gaps in ``tops`` / ``lefts`` may be
    shorter than the nominal stride near the right/bottom edge.

    ``n_h``, ``n_w``, ``total_tiles``, ``stride_h``, ``stride_w`` are derived
    properties exposed for backward compatibility with callers that only
    read them. To compute a tile position by index, prefer
    ``grid.tops[row]`` / ``grid.lefts[col]`` over ``row * grid.stride_h`` —
    the latter is wrong for the last clamped tile.
    """

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

    @property
    def stride_h(self) -> int:
        """Nominal vertical stride (gap between first two rows; ``tile_h`` if only one row)."""
        return self.tops[1] - self.tops[0] if len(self.tops) > 1 else self.tile_h

    @property
    def stride_w(self) -> int:
        """Nominal horizontal stride (gap between first two cols; ``tile_w`` if only one col)."""
        return self.lefts[1] - self.lefts[0] if len(self.lefts) > 1 else self.tile_w


def axis_positions(length: int, tile: int, stride: int) -> tuple[int, ...]:
    """Return tile start positions covering ``[0, length - tile]``.

    Walks at the given ``stride`` from 0 and appends positions while the tile
    still fits inside ``length``. If the last walked position is not exactly
    ``length - tile``, appends one final position clamped to the right edge.
    Single-position result if ``tile >= length``.

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        stride: Nominal stride between consecutive tiles.

    Returns:
        Strictly increasing tuple of start positions.
    """
    if tile <= 0 or stride <= 0:
        msg = f"tile and stride must be positive (got tile={tile}, stride={stride})"
        raise ValueError(msg)
    if tile >= length:
        return (0,)

    positions: list[int] = []
    p = 0
    while p + tile <= length:
        positions.append(p)
        p += stride
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


def compute_tile_grid(
    h: int,
    w: int,
    resolution_scale: int,
    overlap: float = 0.5,
    tile_hw: tuple[int, int] | None = None,
) -> TileGrid:
    """Compute tile geometry with configurable overlap.

    By default, tile size is ``(h // resolution_scale, w // resolution_scale)``.
    Pass ``tile_hw`` to override with explicit tile dimensions (e.g. derived
    from a fixed base resolution rather than the source frame size).

    The grid covers the full frame: if the nominal stride does not divide
    evenly into ``(h - tile_h)`` / ``(w - tile_w)``, the last tile in each
    axis is clamped to the right/bottom edge. ``stitch_tiles_hanning``
    handles the resulting non-uniform overlap correctly via Hanning-window
    normalisation.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        resolution_scale: Divisor for tile size when ``tile_hw`` is ``None``.
        overlap: Fraction of tile overlap in ``[0, 1)``. Default ``0.5`` (50%).
        tile_hw: Explicit ``(tile_h, tile_w)`` override. When set,
            ``resolution_scale`` is ignored for tile sizing.

    Returns:
        ``TileGrid`` with tile sizes and per-axis tile start positions.
    """
    if tile_hw is None:
        tile_h = h // resolution_scale
        tile_w = w // resolution_scale
    else:
        tile_h, tile_w = tile_hw
    stride_h = max(1, int(tile_h * (1.0 - overlap)))
    stride_w = max(1, int(tile_w * (1.0 - overlap)))

    tops = axis_positions(h, tile_h, stride_h)
    lefts = axis_positions(w, tile_w, stride_w)
    return TileGrid(tile_h=tile_h, tile_w=tile_w, tops=tops, lefts=lefts)


def tile_origin_from_index(grid: TileGrid, tile_index: int) -> tuple[int, int]:
    """Recover the grid-aligned ``(top, left)`` pixel origin of a tile from its index.

    Tiles are numbered row-major: ``tile_index = row * n_w + col``.

    Args:
        grid: Tile grid parameters from :func:`compute_tile_grid`.
        tile_index: Row-major tile index in ``[0, grid.total_tiles)``.

    Returns:
        ``(top, left)`` pixel offset of the tile within the (unscaled) frame.

    Raises:
        ValueError: If ``tile_index`` is outside ``[0, grid.total_tiles)``.
    """
    if not 0 <= tile_index < grid.total_tiles:
        msg = f"tile_index {tile_index} out of range [0, {grid.total_tiles})"
        raise ValueError(msg)
    row, col = divmod(tile_index, grid.n_w)
    return grid.tops[row], grid.lefts[col]


def extract_all_tiles(
    video: torch.Tensor,
    grid: TileGrid,
) -> list[torch.Tensor]:
    """Extract all spatial tiles from a video tensor.

    Args:
        video: ``[T, C, H, W]`` tensor.
        grid: Tile grid parameters from ``compute_tile_grid``.

    Returns:
        List of ``[T, C, tile_h, tile_w]`` tensors in row-major order
        (``tops`` x ``lefts``).
    """
    tiles: list[torch.Tensor] = []
    for top in grid.tops:
        for left in grid.lefts:
            tile = video[:, :, top : top + grid.tile_h, left : left + grid.tile_w]
            tiles.append(tile)
    return tiles


def _normalize_blend(acc: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Divide accumulated, Hann-weighted tile values by their accumulated weight.

    Every destination pixel inside the tile grid's covered region has a strictly positive
    weight (a Hann window with non-zero endpoints, see :func:`hanning_window_2d`), even at a
    tile's extreme corner -- for a 512-wide tile that corner weight is about ``6e-10``, still
    far above the float32 subnormal floor, so dividing by the *exact* accumulated weight is
    numerically fine (the same tiny factor scales both the accumulated pixel value and the
    weight, and cancels). Clamping the weight to a floor like ``1e-6`` before dividing -- the
    previous behaviour here -- replaces that tiny-but-correct weight with a much larger one,
    scaling a valid corner pixel down by orders of magnitude (a flat white tile's corner came
    out almost black). The only pixels that may have exactly zero weight are ones genuinely
    outside every tile (a frame the grid does not fully cover, which should not happen for a
    grid built by :func:`compute_tile_grid` / :func:`compute_tile_grid_even`, but is handled
    defensively rather than assumed away): zero those explicitly instead of dividing by a
    fabricated floor.

    Args:
        acc: Accumulated ``window * tile_value`` sum, any leading dims, trailing ``[H, W]``.
        weight: Accumulated ``window`` sum, broadcastable to ``acc``'s trailing ``[H, W]``.

    Returns:
        ``acc / weight`` where ``weight > 0``, else ``0``.
    """
    covered = weight > 0
    safe_weight = torch.where(covered, weight, torch.ones_like(weight))
    return torch.where(covered, acc / safe_weight, torch.zeros_like(acc))


def hanning_window_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Hanning window with non-zero endpoints.

    Uses ``hann_window(n + 2)[1:-1]`` to avoid exact zeros at boundaries,
    ensuring non-zero weight where only one tile contributes.

    Args:
        h: Window height.
        w: Window width.
        device: Target device.

    Returns:
        ``[h, w]`` float tensor with values in ``(0, 1]``.
    """
    wy = torch.hann_window(h + 2, device=device)[1:-1]
    wx = torch.hann_window(w + 2, device=device)[1:-1]
    return wy[:, None] * wx[None, :]


def stitch_tiles_hanning(
    tiles: list[torch.Tensor],
    grid: TileGrid,
    original_h: int,
    original_w: int,
    scale: int = 1,
) -> torch.Tensor:
    """Stitch tiles into a full frame using Hanning-window weighted blending.

    Each tile is multiplied by a 2D Hanning window and accumulated into
    the output canvas. The final result is normalised by the accumulated
    weights so that overlapping regions blend smoothly. The same scheme
    works for non-uniform overlap at the right/bottom edge: in the clamped
    region both ``pred_acc`` and ``weight_acc`` receive more contributions,
    and the per-pixel division cancels it out.

    When ``scale > 1``, tiles are assumed to be at HR resolution
    (i.e. each tile covers ``tile_h * scale x tile_w * scale`` pixels)
    and the output canvas is ``original_h * scale x original_w * scale``.
    Grid positions are scaled accordingly.

    Args:
        tiles: List of ``[C, T, th, tw]`` float tensors in row-major order
            (same order as ``extract_all_tiles``). When ``scale == 1``,
            ``th == grid.tile_h``; when ``scale > 1``, ``th == grid.tile_h * scale``.
        grid: Tile grid parameters (at LQ / original resolution).
        original_h: LQ frame height.
        original_w: LQ frame width.
        scale: Upscale factor. Output resolution is
            ``(original_h * scale, original_w * scale)``.

    Returns:
        ``[C, T, original_h * scale, original_w * scale]`` float tensor.
    """
    first = tiles[0]
    c, t = first.shape[0], first.shape[1]
    hr_tile_h, hr_tile_w = first.shape[2], first.shape[3]
    device = first.device

    window = hanning_window_2d(hr_tile_h, hr_tile_w, device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, hr_tile_h, hr_tile_w]

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

    return _normalize_blend(pred_acc, weight_acc)


def axis_positions_even(
    length: int, tile: int, min_overlap: float, snap: int
) -> tuple[int, ...]:
    """Return evenly distributed, ``snap``-aligned tile start positions.

    Uses the minimal position count whose uniform stride keeps the tile
    overlap at or above ``min_overlap``, then spreads the positions evenly
    over ``[0, length - tile]`` in integer ``snap`` units. When ``length`` or
    ``tile`` is not ``snap``-aligned the same layout is computed at pixel
    precision (``snap=1``) — the latent-grid guard downstream still enforces
    alignment where it actually matters (the LU path).

    The legacy :func:`axis_positions` walks a fixed stride and clamps the last
    tile to the edge, piling all layout slack into the final step (width 976 /
    tile 384 / stride 288 -> ``(0, 288, 576, 592)``); this layout distributes
    the minimal number of tiles uniformly instead.

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        min_overlap: Overlap floor as a fraction of ``tile`` in ``[0, 1)``.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        Strictly increasing positions covering ``[0, length - tile]``.

    Raises:
        ValueError: On non-positive ``tile``/``snap`` or ``min_overlap``
            outside ``[0, 1)``.
    """
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
    """Build a :class:`TileGrid` with the even per-axis layout.

    Drop-in for ``compute_tile_grid(..., tile_hw=...)``: same ``TileGrid``
    contract (``tops``/``lefts`` are authoritative), only the position layout
    differs — see :func:`axis_positions_even`.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        tile_hw: Explicit ``(tile_h, tile_w)``.
        min_overlap: Overlap floor as a fraction of the tile size.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        ``TileGrid`` with evenly distributed, aligned tile positions.
    """
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
    """Pick the trained base ``(H, W)`` whose aspect ratio is closest to ``w / h``.

    Args:
        h: Source frame height.
        w: Source frame width.
        visual_size: Resolution-registry key (only ``512`` is registered by default).
        resolutions: ``visual_size -> [(H, W), ...]`` registry (default :data:`RESOLUTIONS`).

    Raises:
        ValueError: If ``visual_size`` is not a key of ``resolutions``.
    """
    if visual_size not in resolutions:
        raise ValueError(
            f"Unsupported SR visual_size={visual_size}; known sizes: {sorted(resolutions)}"
        )
    ratio = w / h if h else 1.0
    return min(resolutions[visual_size], key=lambda hw: abs(hw[1] / hw[0] - ratio))


def tile_geometry(  # noqa: PLR0913
    h: int,
    w: int,
    visual_size: int,
    scale: int,
    overlap: float,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
    grid_mode: TileGridMode = "even",
    grid_min_overlap: float = 0.20,
    resolutions: Mapping[int, Sequence[tuple[int, int]]] = RESOLUTIONS,
) -> tuple[tuple[int, int], tuple[int, int], TileGrid]:
    """Resolve ``(base_hw, tile_hw, pixel_tile_grid)`` for a frame at a tiling scale.

    Args:
        h: Source frame height.
        w: Source frame width.
        visual_size: Resolution-registry key.
        scale: Integer tiling scale; must divide the chosen base resolution.
        overlap: Nominal overlap of the ``"legacy"`` grid.
        spatial_factor: VAE spatial factor; the ``"even"`` grid snaps positions to it.
        grid_mode: ``"even"`` (minimal uniform layout) or anything else for the legacy walk.
        grid_min_overlap: Overlap floor of the ``"even"`` grid.
        resolutions: Resolution registry (default :data:`RESOLUTIONS`).

    Returns:
        ``((base_h, base_w), (tile_h, tile_w), grid)``.

    Raises:
        ValueError: On an unknown ``visual_size`` or a ``scale`` that does not divide the base.
    """
    base_h, base_w = closest_base_resolution(h, w, visual_size, resolutions)
    if base_h % scale or base_w % scale:
        raise ValueError(
            f"resolution_scale={scale} does not divide the base resolution {base_h}x{base_w} exactly"
        )
    tile_hw = (base_h // scale, base_w // scale)
    if grid_mode == "even":
        grid = compute_tile_grid_even(h, w, tile_hw, grid_min_overlap, spatial_factor)
    else:
        grid = compute_tile_grid(h, w, scale, overlap, tile_hw=tile_hw)
    return (base_h, base_w), tile_hw, grid


def upsample_tiles_to_base(
    raw_tiles: list[torch.Tensor], base_h: int, base_w: int
) -> list[torch.Tensor]:
    """Bilinearly resize ``[T, C, h, w]`` tiles to ``base_h x base_w`` and return ``[T, H, W, C]``."""
    return [
        functional.interpolate(
            tile.float(), size=(base_h, base_w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        for tile in raw_tiles
    ]


def resolve_scale_request(scale: float) -> tuple[int, float]:
    """Map a requested total resolution scale to ``(tiling_scale, pre_upscale)``.

    Fractional totals are decomposed into a pixel-space pre-upscale times an
    integer tiling scale (``2.25 = 1.125 x 2``) so the regular integer-scale
    tile machinery runs unchanged on the slightly enlarged source.

    Args:
        scale: The requested total upscale — ``2``, ``4`` or ``2.25``.

    Returns:
        ``(tiling_scale, pre_upscale)``; ``pre_upscale`` is ``1.0`` for the
        integer scales.
    """
    requested = float(scale)
    if requested == 2.25:  # the one supported fractional total
        return 2, 1.125
    if requested in (2.0, 4.0):
        return int(requested), 1.0
    raise ValueError("SR supports total scales 2, 4, and 2.25")


def pre_upscale_video(
    video: torch.Tensor, factor: float, spatial_multiple: int
) -> torch.Tensor:
    """Bilinear-upscale a ``[T, C, H, W]`` uint8 video by ``factor`` in pixel space.

    Target dims are rounded to the nearest multiple of ``spatial_multiple``
    (the VAE spatial factor) so the whole-video encode and the latent tile
    grid stay integer-aligned (``latent_tile_grid_from_pixel_grid`` rejects
    unaligned grids). For sources whose scaled dims already land on the
    factor (512x768 x1.125 -> 576x864) the rounding is a no-op and the total
    scale is exact.

    Args:
        video: ``[T, C, H, W]`` uint8 source video.
        factor: Pixel upscale factor (> 1).
        spatial_multiple: VAE spatial factor to align the target dims to.

    Returns:
        ``[T, C, H', W']`` uint8 video with ``H' ~= H * factor`` aligned.
    """
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
    """Edge-replicate-pad a ``[T, C, H, W]`` video's bottom/right so ``H`` and ``W`` are both
    multiples of ``spatial_factor``, for whole-video KVAE encoding.

    The KVAE halves H and W four times (``spatial_factor == 16``); an input whose dims are not
    a multiple of 16 fails partway through that cascade instead of at a clear boundary (e.g. a
    1080-tall frame halves to 540, 270, 135 -- odd, so the fourth halving has no integer
    result). :func:`pre_upscale_video` already lands on a 16-aligned size for the one scale
    that resizes pixels (2.25); integer scales (2, 4) run no pixel resize at all, so without
    this padding only they would hit the misalignment. Padding (not resizing) keeps every
    source pixel and is simply cropped back off the far end of the result once it is known
    (:func:`crop_to_hw`) -- the caller is expected to record the pre-pad ``(H, W)`` this
    function returns and crop the SR output down to ``(H * scale, W * scale)`` afterward.

    Args:
        video: ``[T, C, H, W]`` source video (any dtype ``F.pad`` supports in ``replicate``
            mode, i.e. floating or integer for the 4D case -- uint8 included).
        spatial_factor: The alignment multiple (the VAE's spatial compression factor).

    Returns:
        ``(padded_video, (H, W))`` -- the padded video and the original, unpadded size to
        restore after upscaling.
    """
    if video.ndim != 4:
        raise ValueError(f"video must have rank 4 [T,C,H,W], got {tuple(video.shape)}")
    height, width = video.shape[-2:]
    pad_h = (-height) % spatial_factor
    pad_w = (-width) % spatial_factor
    if pad_h == 0 and pad_w == 0:
        return video, (height, width)
    padded = functional.pad(video, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def crop_to_hw(video: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Crop a ``[..., H, W]`` tensor's trailing two dims down to ``hw`` from the top-left.

    The inverse half of :func:`pad_to_spatial_factor`: that function only ever pads the
    bottom/right edges, so cropping the same corner back off restores exactly the original
    (pre-pad) content, scaled by whatever the SR run did to every pixel equally.
    """
    height, width = hw
    return video[..., :height, :width]


__all__ = [
    "RESOLUTIONS",
    "VAE_SPATIAL_FACTOR",
    "VAE_TEMPORAL_FACTOR",
    "TileGrid",
    "TileGridMode",
    "axis_positions",
    "axis_positions_even",
    "closest_base_resolution",
    "compute_tile_grid",
    "compute_tile_grid_even",
    "crop_to_hw",
    "extract_all_tiles",
    "hanning_window_2d",
    "pad_to_spatial_factor",
    "latent_tile_grid_from_pixel_grid",
    "pre_upscale_video",
    "resolve_scale_request",
    "stitch_tiles_hanning",
    "tile_geometry",
    "tile_origin_from_index",
    "upsample_tiles_to_base",
]
