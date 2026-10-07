# SPDX-License-Identifier: Apache-2.0
"""Tiled spatial decoding must preserve each tile's original image boundaries.

Before the fix, every rank decoded the complete tile with spatial wrappers
enabled. Its halo therefore came from another copy of the tile instead of the
neighboring spatial shard, changing decoded pixels at the image boundaries.
Short tiles and full inputs must also decode when there are more ranks than
rows, and leave spatial decoding enabled for the next request.
"""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.test.test_utils import CustomTestCase

# Native diffusion imports require this lane's Diffusers dependencies.
# The regression itself uses CPU tensors and real Gloo process groups.
register_cuda_ci(
    est_time=300, stage="base-b", runner_config="diffusion-unit-1-gpu-h100"
)


def _check_tiled_spatial_decode(
    rank, rendezvous, world_size=2, latent_shapes=((15, 13),)
):
    # Each spawned worker selects CPU before importing the diffusion runtime.
    os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cpu"
    torch.set_num_threads(1)

    from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
    from sglang.multimodal_gen.runtime.distributed import group_coordinator
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_decode_parallel_world_size,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.parallel_conv import (
        disable_spatial_parallel_decode,
        spatial_parallel_decode_disabled,
    )
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="gloo",
        timeout=60,
    )
    try:
        initialize_model_parallel(
            tensor_parallel_degree=world_size,
            sequence_parallel_degree=1,
            backend="gloo",
        )
        assert get_decode_parallel_world_size() == world_size
        config = FluxVAEConfig()
        config.use_parallel_decode = True
        config.parallel_decode_mode = "spatial_shard"
        config.update_model_arch(
            {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "sample_size": 16,
                "block_out_channels": (32, 32),
                "layers_per_block": 1,
                "norm_num_groups": 32,
                "act_fn": "silu",
                "down_block_types": ("DownEncoderBlock2D",) * 2,
                "up_block_types": ("UpDecoderBlock2D",) * 2,
                "mid_block_add_attention": True,
                "use_quant_conv": True,
                "use_post_quant_conv": True,
            }
        )
        torch.manual_seed(17)
        vae = AutoencoderKL(config).eval()
        assert vae._spatial_parallel_decode_enabled
        vae.enable_tiling()
        # Select real Gloo collectives instead of the optional AMX shared-memory
        # transport; neither the spatial algorithm nor any collective is mocked.
        with (
            patch.object(group_coordinator, "is_shm_available", return_value=False),
            torch.no_grad(),
        ):
            for height, width in latent_shapes:
                latent = torch.randn(1, 4, height, width)
                with disable_spatial_parallel_decode():
                    reference = vae.decode(latent)
                actual = vae.decode(latent)
                assert not spatial_parallel_decode_disabled(), (
                    "A small-input fallback left spatial decoding disabled"
                )
                difference = (actual - reference).abs()
                print(
                    f"world={world_size} rank={rank} latent={height}x{width} "
                    f"shape={tuple(actual.shape)} "
                    f"max_abs={difference.max().item():.9g} "
                    f"bottom_max_abs={difference[..., -4:, :].max().item():.9g}",
                    flush=True,
                )
                torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


class TestTiledSpatialDecode(CustomTestCase):
    def test_two_ranks_match_serial_tiled_decode(self):
        # Fresh workers own all distributed state and release every group on
        # success or assertion failure; no state leaks into other CPU tests.
        with TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            mp.spawn(_check_tiled_spatial_decode, args=(rendezvous,), nprocs=2)

    def test_four_ranks_handle_small_tiles_and_restore_spatial_decode(self):
        # H14 creates a final two-row tile, shorter than the decode group.
        # H2xW3 exercises the non-tiled small full-input fallback. The final
        # normal decode also checks that the previous calls left no stale
        # partition metadata or disabled spatial context behind.
        with TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            mp.spawn(
                _check_tiled_spatial_decode,
                args=(rendezvous, 4, ((14, 13), (2, 3), (12, 9))),
                nprocs=4,
            )


if __name__ == "__main__":
    unittest.main()
