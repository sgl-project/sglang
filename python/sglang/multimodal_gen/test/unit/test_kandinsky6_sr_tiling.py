# SPDX-License-Identifier: Apache-2.0
"""Tile blending and initial latent precision."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    plan_tiles,
    prepare_tile_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    compute_tile_grid_even,
    stitch_tiles_hanning,
)


@pytest.mark.parametrize(
    "height,width,tile_hw",
    [
        (512, 512, (512, 512)),
        (512, 768, (512, 768)),
        (768, 512, (768, 512)),
        (768, 1536, (512, 768)),
    ],
)
@pytest.mark.parametrize("value", [0.0, 1.0, 255.0])
def test_hann_blending_preserves_constant_color_including_corners(
    height, width, tile_hw, value
):
    # tiny but nonzero Hann corner weights must not be clamped before normalization
    grid = compute_tile_grid_even(height, width, tile_hw, 0.2, 16)
    tiles = [torch.full((3, 1, *tile_hw), value) for _ in range(grid.total_tiles)]
    result = stitch_tiles_hanning(tiles, grid, height, width, scale=1)
    torch.testing.assert_close(
        result, torch.full_like(result, value), rtol=1e-5, atol=1e-2
    )


def test_uncovered_pixels_are_zero_not_nan():
    grid = compute_tile_grid_even(64, 64, (64, 64), 0.2, 16)
    tile = torch.full((3, 1, 64, 64), 100.0)
    result = stitch_tiles_hanning([tile], grid, 64, 128, scale=1)
    assert result.shape == (3, 1, 64, 128)
    torch.testing.assert_close(result[..., :64], tile, rtol=1e-5, atol=1e-2)
    assert torch.equal(result[..., 64:], torch.zeros_like(result[..., 64:]))


@pytest.mark.parametrize("dit_dtype", [torch.bfloat16, torch.float32, None])
def test_initial_latent_uses_dit_dtype_or_preserves_input(dit_dtype):
    dit_spec = DitSpec(
        instruct_type="noise",
        visual_cond=False,
        in_visual_dim=4,
        use_motion_score=False,
        patch_size=(1, 1, 1),
        dtype=dit_dtype,
    )
    spec = SamplingSpec(
        tiling_scale=2,
        tiles_batch_size=1,
        seed=1,
        num_steps=5,
        is_piflow=False,
        tile_min_overlap=0.2,
        visual_size=512,
        scale_factor=(1.0, 1.0, 1.0),
        lq_noise_scale=0.7,
        lq_noise_type="linear",
        lq_channel_noise_scale=0.0,
        cap_noise_timestep=False,
    )
    tile = torch.randn(4, 4, 4, 4, dtype=torch.float64)
    plan = plan_tiles(
        frame_hw=(64, 64),
        visual_size=512,
        tiling_scale=2,
        tile_min_overlap=0.2,
        resolutions={512: [(128, 128)]},
    )
    result = prepare_tile_latents(
        tile,
        plan,
        tile_encoder=lambda tile: tile.permute(0, 2, 3, 1),
        latent_path=True,
        dit_spec=dit_spec,
        spec=spec,
        device=torch.device("cpu"),
    )[0]
    assert result.dtype == (dit_dtype or tile.dtype)
