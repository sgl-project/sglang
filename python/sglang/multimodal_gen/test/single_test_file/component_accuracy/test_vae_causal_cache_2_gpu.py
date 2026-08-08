"""Two-GPU checks for the VAE temporal causal cache under height sharding.

The temporal cache is rank-local: the decoder splits the height before any
convolution runs, so each rank only ever caches its own slice and the frames it
carries across chunks are never exchanged. What keeps the ranks consistent is
the per-conv halo exchange, which is orthogonal. These tests pin that down by
requiring a sharded decode to agree with the same decode run unsharded.

Tiny randomly-initialised VAEs -- no checkpoint is downloaded.
"""

import os

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.causal_conv3d_cache import CausalConv3d
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelCausalConv3d,
    disable_spatial_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

# Height must divide evenly across the ranks and stay above the halo width.
LATENT_SHAPE = (1, 4, 3, 16, 12)


@pytest.fixture(scope="module")
def decode_group():
    ensure_distributed_env_defaults()
    launched_ranks = int(os.environ.get("WORLD_SIZE", "1"))
    if not model_parallel_is_initialized():
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=launched_ranks,
            cfg_degree=1,
            # sp_size must factor into ulysses x ring.
            ulysses_degree=launched_ranks,
            ring_degree=1,
            dp_size=1,
        )
    world_size = get_decode_parallel_world_size()
    if world_size < 2:
        pytest.skip(f"needs a decode group of at least 2 ranks, got {world_size}")
    return world_size


def _tiny_sharded_wan_vae() -> AutoencoderKLWan:
    arch = WanVAEArchConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2, 2),
        num_res_blocks=1,
        temperal_downsample=(False, True),
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
        scale_factor_temporal=2,
        scale_factor_spatial=4,
    )
    config = WanVAEConfig(arch_config=arch)
    config.load_encoder = False
    config.use_parallel_encode = False
    config.use_parallel_decode = True
    config.parallel_decode_mode = "spatial_shard"

    torch.manual_seed(0)
    vae = AutoencoderKLWan(config).eval().cuda().to(torch.bfloat16)
    # Same weights on every rank: the ranks must be comparing the same model.
    for param in vae.parameters():
        dist.broadcast(param.data, src=0)
    return vae


def _latents() -> torch.Tensor:
    torch.manual_seed(1234)
    z = torch.randn(LATENT_SHAPE).cuda().to(torch.bfloat16)
    dist.broadcast(z, src=0)
    return z


def test_decoder_is_actually_height_sharded(decode_group):
    """Guard the premise of the other tests: the shard has to be installed."""
    vae = _tiny_sharded_wan_vae()
    sharded = [
        m for m in vae.decoder.modules() if isinstance(m, SpatialParallelCausalConv3d)
    ]
    assert sharded, "decoder should have been built with height-sharded convs"


def test_sharded_decode_matches_unsharded(decode_group):
    vae = _tiny_sharded_wan_vae()
    z = _latents()

    with torch.no_grad():
        sharded = vae.decode(z)
        with disable_spatial_parallel_decode():
            reference = vae.decode(z)

    assert sharded.shape == reference.shape
    torch.testing.assert_close(sharded.float(), reference.float(), rtol=0, atol=2e-3)


def test_every_rank_gathers_the_same_output(decode_group):
    vae = _tiny_sharded_wan_vae()
    z = _latents()
    with torch.no_grad():
        decoded = vae.decode(z)

    gathered = [torch.empty_like(decoded) for _ in range(decode_group)]
    dist.all_gather(gathered, decoded.contiguous())
    for rank, other in enumerate(gathered):
        torch.testing.assert_close(
            decoded.float(), other.float(), rtol=0, atol=0, msg=f"rank {rank} differs"
        )


def test_cached_frames_are_rank_local(decode_group):
    """Each rank must cache its own height slice, not the full-height tensor."""
    vae = _tiny_sharded_wan_vae()
    z = _latents()

    heights = []
    conv = vae.decoder.conv_in
    real_consume = conv._consume_cache

    def spy(x, cache):
        out = real_consume(x, cache)
        heights.append((x.shape[-2], cache.get(conv.cache_key).shape[-2]))
        return out

    conv._consume_cache = spy
    with torch.no_grad():
        vae.decode(z)

    assert heights, "conv_in should have consumed the cache"
    local_height = LATENT_SHAPE[-2] // decode_group
    for input_height, cached_height in heights:
        assert input_height == local_height
        assert cached_height == local_height


def test_toggling_the_shard_does_not_leak_cache_shapes(decode_group):
    """Switching the shard off changes the height a cached tail would have."""
    vae = _tiny_sharded_wan_vae()
    z = _latents()
    with torch.no_grad():
        # Interleaved on purpose: a cache surviving across the switch would be
        # the wrong height and raise on the next concatenation.
        vae.decode(z)
        with disable_spatial_parallel_decode():
            vae.decode(z)
        again = vae.decode(z)
        with disable_spatial_parallel_decode():
            reference = vae.decode(z)

    torch.testing.assert_close(again.float(), reference.float(), rtol=0, atol=2e-3)


def test_pointwise_convs_stay_off_the_shard(decode_group):
    """1x1x1 convs need no halo, so they must not be sharded or cached."""
    vae = _tiny_sharded_wan_vae()
    for name in ("quant_conv", "post_quant_conv"):
        conv = getattr(vae, name)
        assert isinstance(conv, CausalConv3d)
        assert not isinstance(conv, SpatialParallelCausalConv3d)
        assert conv.cache_frames == 0
