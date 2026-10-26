# SPDX-License-Identifier: Apache-2.0
"""ParallelTiledVAE tiling: SP-group communication under TP x SP, seam blending."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.runtime.models.vaes import common


class _IdentityVAE(common.ParallelTiledVAE):
    def _encode(self, x):
        return x

    def _decode(self, z):
        return z


def _decode_in_sp_group(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        group = groups[rank % 2]
        coordinator = SimpleNamespace(
            world_size=2,
            rank_in_group=rank // 2,
            device_group=group,
            cpu_group=group,
        )
        config = VAEConfig(
            arch_config=VAEArchConfig(
                temporal_compression_ratio=1, spatial_compression_ratio=1
            ),
            tile_sample_min_height=2,
            tile_sample_min_width=2,
            tile_sample_stride_height=1,
            tile_sample_stride_width=1,
            tile_sample_min_num_frames=4,
            tile_sample_stride_num_frames=3,
        )
        vae = _IdentityVAE(config)
        # Different content in each group detects cross-group contamination.
        z = torch.arange(45, dtype=torch.float32).reshape(1, 1, 5, 3, 3)
        z = z + 1000 * (rank % 2)
        expected = vae.tiled_decode(z)
        with patch.object(common, "get_sp_group", return_value=coordinator):
            actual = vae.parallel_tiled_decode(z)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_tiled_decode_uses_its_sp_group(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_decode_in_sp_group, args=(rendezvous,), nprocs=4)


def _blend_per_slice(a, b, blend_extent, dim):
    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    for i in range(blend_extent):
        b.select(dim, i).copy_(
            a.select(dim, a.shape[dim] - blend_extent + i) * (1 - i / blend_extent)
            + b.select(dim, i) * (i / blend_extent)
        )
    return b


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [-3, -2, -1])
@pytest.mark.parametrize("blend_extent", [1, 7, 64, 100])
def test_blend_matches_per_slice_loop(dtype, dim, blend_extent):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(1, 3, 9, 70, 66, generator=generator).to(device, dtype)
    b = torch.randn(1, 3, 9, 70, 66, generator=generator).to(device, dtype)
    expected = _blend_per_slice(a, b.clone(), blend_extent, dim)
    actual = common.ParallelTiledVAE._blend(a, b.clone(), blend_extent, dim)
    assert torch.equal(actual, expected)
