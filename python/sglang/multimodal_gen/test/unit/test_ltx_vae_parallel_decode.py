# SPDX-License-Identifier: Apache-2.0
"""Two decode ranks must reproduce the single-rank LTX-2 VAE decode.

The ranks run over gloo on CPU, so this needs no GPU.
"""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
    LTXVideoVAEArchConfig,
    LTXVideoVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.parallel_conv import SpatialParallelConv3d
from sglang.multimodal_gen.runtime.models.vaes import common
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
    AutoencoderKLCausalLTX2Video,
)

WORLD_SIZE = 2


class _DecodeGroup:
    def __init__(self, rank: int):
        self.rank_in_group = rank
        self.world_size = WORLD_SIZE
        self.ranks = list(range(WORLD_SIZE))
        self.device_group = dist.group.WORLD
        self.cpu_group = dist.group.WORLD

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        gathered = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(gathered, x, group=self.device_group)
        return torch.cat(gathered, dim=dim)


def _tiny_vae(**config) -> AutoencoderKLCausalLTX2Video:
    torch.manual_seed(0)
    arch = LTXVideoVAEArchConfig(
        latent_channels=4,
        spatial_compression_ratio=2,
        temporal_compression_ratio=2,
        block_out_channels=[8],
        down_block_types=["LTX2VideoDownBlock3D"],
        spatio_temporal_scaling=[True],
        layers_per_block=[1, 1],
        downsample_type=["spatiotemporal"],
        patch_size=1,
        decoder_block_out_channels=[8],
        decoder_spatio_temporal_scaling=[True],
        decoder_layers_per_block=[1, 1],
        decoder_inject_noise=[False, False],
        upsample_residual=[True],
        upsample_factor=[2],
    )
    vae = AutoencoderKLCausalLTX2Video(LTXVideoVAEConfig(arch_config=arch, **config))
    # 4-row latent tiles on a 3-row stride: an 8x8 latent spans 3x3 tiles
    vae.enable_tiling(
        tile_sample_min_height=8,
        tile_sample_min_width=8,
        tile_sample_min_num_frames=8,
        tile_sample_stride_height=6,
        tile_sample_stride_width=6,
        tile_sample_stride_num_frames=4,
    )
    return vae.eval()


@torch.no_grad()
def _check_decodes(rank: int) -> None:
    reference = _tiny_vae(use_parallel_decode=False)
    auto = _tiny_vae()
    sharded = _tiny_vae(parallel_decode_mode="spatial_shard")
    assert any(isinstance(m, SpatialParallelConv3d) for m in sharded.decoder.modules())
    torch.manual_seed(1)
    z = torch.randn(1, 4, 7, 8, 8)
    errors = {}

    def compare(name, actual, expected):
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        errors[name] = max(
            errors.get(name, 0.0), (actual - expected).abs().max().item()
        )

    # below the auto threshold every rank decodes the whole clip, tiled as on one rank
    compare("tiled", auto.decode(z).sample, reference.decode(z).sample)
    # 4 rows x 8 per rank exceed a 4x4 tile, so the ranks split the tiles instead
    compare("tiled_shard", sharded.decode(z).sample, reference.decode(z).sample)
    # they fit a 6x6 tile, so the height split replaces tiling
    sharded.enable_tiling(
        tile_sample_min_height=12,
        tile_sample_min_width=12,
        tile_sample_stride_height=10,
        tile_sample_stride_width=10,
    )
    reference.use_tiling = False
    compare("height_split", sharded.decode(z).sample, reference.decode(z).sample)
    reference.use_framewise_decoding = sharded.use_framewise_decoding = True
    compare("framewise", sharded.decode(z).sample, reference.decode(z).sample)
    reference_cache, cache = {}, {}
    for chunk in z.split(3, dim=2):
        compare(
            "chunk",
            sharded.decode_chunk(chunk, cache),
            reference.decode_chunk(chunk, reference_cache),
        )
    mismatched = {name: err for name, err in errors.items() if err > 1e-5}
    assert not mismatched, f"rank {rank}: max abs error vs one rank {mismatched}"


def _decode_on_two_ranks(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=60),
    )
    parallel_state._VAE_DECODE = _DecodeGroup(rank)
    try:
        with patch.object(common, "model_parallel_is_initialized", return_value=True):
            _check_decodes(rank)
    finally:
        parallel_state._VAE_DECODE = None
        dist.destroy_process_group()


def test_two_decode_ranks_match_one_rank(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_decode_on_two_ranks, args=(rendezvous,), nprocs=WORLD_SIZE)
