# SPDX-License-Identifier: Apache-2.0
"""Tile-grid / Hann-stitch pure-function tests for Kandinsky 6 video SR.

SGLang PR #2 review comment on ``tiling.py``: blanket-clamping the accumulated Hann weight to
``min=1e-6`` before dividing corrupts tiny-but-valid weights (a 512-wide tile's corner is about
``6e-10``), turning a valid pixel almost black instead of normalizing it correctly. These tests
pin the fix (:func:`.tiling._normalize_blend`) at realistic tile sizes.
"""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    compute_tile_grid_even,
    hanning_window_2d,
    stitch_tiles_hanning,
)

# The trained SR base resolutions (tiling.RESOLUTIONS[512]): a single-tile frame at each one
# exercises the true worst-case corner weight the released checkpoints actually produce tiles
# at.
REALISTIC_TILE_SIZES = [(512, 512), (512, 768), (768, 512)]


def test_corner_hann_weight_is_far_above_the_old_clamp_floor():
    """Sanity check for the bug report itself: the corner weight really is tiny, but not
    anywhere near small enough to excuse clamping it to 1e-6."""
    window = hanning_window_2d(512, 768, torch.device("cpu"))
    corner = window[0, 0].item()
    assert 0 < corner < 1e-6
    assert corner > torch.finfo(torch.float32).tiny  # nowhere near underflow


def test_constant_color_tile_stays_uniform_after_stitching():
    """A single tile covering the whole frame: every destination pixel (including all four
    corners) is only ever weighted by that one tile's Hann window, so the blended-and-
    normalized result must reproduce the constant value exactly, corners included. Fails
    on the pre-fix ``clamp(min=1e-6)``, which divides the (tiny, valid) corner accumulator
    by a floor many orders of magnitude larger than the true weight and darkens it."""
    for tile_h, tile_w in REALISTIC_TILE_SIZES:
        grid = compute_tile_grid_even(tile_h, tile_w, (tile_h, tile_w), 0.2, 16)
        for value in (0.0, 1.0, 255.0):
            constant = torch.full((3, 2, tile_h, tile_w), value)
            out = stitch_tiles_hanning([constant], grid, tile_h, tile_w, scale=1)
            assert torch.allclose(out, constant, atol=1e-2), (tile_h, tile_w, value)
            corner = out[0, 0, 0, 0].item()
            assert abs(corner - value) < 1e-2, (
                f"corner darkened: expected {value}, got {corner} for tile "
                f"{tile_h}x{tile_w}"
            )


def test_constant_color_tile_stays_uniform_with_real_overlapping_tiles():
    """Same invariant, but with the actual multi-tile overlapping grid a larger-than-base
    frame produces (not just a single tile covering the whole canvas): every pixel, including
    ones near a tile's own edge that only a single tile's low-weight region covers, must come
    back at the constant input value."""
    height, width = 768, 1536  # two 768x768-ish base tiles side by side, 20% overlap
    tile_h, tile_w = 512, 768
    grid = compute_tile_grid_even(height, width, (tile_h, tile_w), 0.2, 16)
    assert (
        grid.total_tiles > 1
    )  # exercise real multi-tile blending, not the trivial case
    value = 200.0
    tiles = [torch.full((3, 1, tile_h, tile_w), value) for _ in range(grid.total_tiles)]
    out = stitch_tiles_hanning(tiles, grid, height, width, scale=1)
    assert torch.allclose(out, torch.full_like(out, value), atol=1e-2)


def test_stitch_handles_true_zero_coverage_without_nan_or_crashing():
    """A canvas the tile grid does not fully cover (pathological / defensive case: this
    should not happen for a grid built by ``compute_tile_grid`` / ``compute_tile_grid_even``,
    but the fix must not assume it away) gets exactly zero at the uncovered pixels, not NaN
    from a 0/0 division."""
    tile_h, tile_w = 64, 64
    grid = compute_tile_grid_even(tile_h, tile_w, (tile_h, tile_w), 0.2, 16)
    tile = torch.full((3, 1, tile_h, tile_w), 100.0)
    # Stitch into a canvas twice as wide as what the grid actually covers: the right half
    # never receives any tile contribution.
    out = stitch_tiles_hanning([tile], grid, tile_h, tile_w * 2, scale=1)
    assert out.shape[-1] == tile_w * 2 * 1
    covered, uncovered = out[..., :tile_w], out[..., tile_w:]
    assert torch.allclose(covered, tile, atol=1e-2)
    assert torch.equal(uncovered, torch.zeros_like(uncovered))
    assert torch.isfinite(out).all()
