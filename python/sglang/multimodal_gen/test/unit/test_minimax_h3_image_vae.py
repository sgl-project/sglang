# SPDX-License-Identifier: Apache-2.0
"""Image VAE tests.

Tiled decode equals untiled decode, and posterior sampling without a generator
draws from the global RNG.
"""

import os
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

from sglang.kernels.ops import diffusion as diffusion_ops
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3ImageVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
    maybe_optimize_minimax_h3_vae,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKLLegacy,
    DiagonalGaussianDistribution,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline_config import (
    normalized_args,
)

TINY_VIT = {
    "dim_head": 16,
    "ffn_activation_fn": "silu",
    "ffn_use_gated": True,
    "heads": 2,
    "norm_affine": True,
    "norm_type": "rms_norm",
    "num_layers": 2,
    "qk_norm_affine": False,
    "qk_norm_type": "rms_norm",
    "rope_dim_ratio": 0.5,
    "rope_theta": 100.0,
}


def tiny_legacy(**overrides):
    kwargs = dict(
        ch=32,
        ch_mult=[1] * 6,
        embed_dim=24,
        z_channels=24,
        use_3d_conv=True,
        space_down=[2, 2, 2, 2, 1, 1],
        time_down=[1, 2, 2, 1, 1, 1],
        padding_mode="reflect",
        use_t_isolated_gn=True,
        use_vit_decoder=True,
        vit_decoder_2d=True,
        vit_decoder_kwargs=TINY_VIT,
        encoder_tiling=False,
        decoder_tiling=False,
        token_drop=0,
    )
    kwargs.update(overrides)
    return AutoencoderKLLegacy(**kwargs).eval()


@pytest.fixture
def tiny_image(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_USE_FUSED_NORM", "false")
    return _tiny_image()


def _tiny_image():
    config = MiniMaxH3VideoVAEConfig(use_tiling=False)
    config.update_model_arch(
        {
            "vae_token_drop": 0,
            "vae_encoder_tiling": 0,
            "vit_decoder_2d": True,
            "vit_decoder_kwargs": TINY_VIT,
        }
    )
    with torch.device("meta"):
        model = MiniMaxH3ImageVAE(config)
    tiny = tiny_legacy()
    for name in ("encoder", "decoder", "quant_conv", "post_quant_conv"):
        setattr(model, name, getattr(tiny, name))
    return model.eval()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
def test_layerwise_image_vae_preserves_fused_decode(tiny_image, monkeypatch):
    model = maybe_optimize_minimax_h3_vae(tiny_image.cuda())
    model.prepare_decoder_autocast_weights(torch.bfloat16)
    with torch.no_grad():
        for block in model.decoder.transformer_blocks:
            block.scale1.normal_(0, 0.1)
            block.scale2.normal_(0, 0.1)
    latents = [torch.randn(1, 24, 1, 4, 4, device="cuda") for _ in range(2)]
    with set_forward_context(0, None), use_vae_fast_path(model, True):
        expected = [model.decode_image_raw(x) for x in latents]
        model.configure_layerwise_offload(
            normalized_args(layerwise_prefetch_size={"video_vae": 1}),
            component_name="video_vae",
        )
        with patch.object(
            diffusion_ops,
            "h3_vae_scale_add_rmsnorm",
            wraps=diffusion_ops.h3_vae_scale_add_rmsnorm,
        ) as scale_add:
            for _ in range(2):
                for latent, reference in zip(latents, expected):
                    torch.testing.assert_close(
                        model.decode_image_raw(latent), reference, rtol=0, atol=0
                    )
            assert scale_add.call_count == 12
    for manager in model.layerwise_offload_managers:
        manager.release_all()


def test_local_tiling_decode_matches_untiled(tiny_image):
    tiny_image.setup_forward(
        encoder_tiling=True,
        decoder_tiling=True,
        parallel_tiling=False,
        tile_size=32,
        tile_overlap_min=16,
        token_drop=0,
    )
    pixels = torch.rand(1, 3, 48, 48)
    moments = tiny_image.tiled_encode(tiny_image.transform(pixels).unsqueeze(2))
    latents = DiagonalGaussianDistribution(moments).mean
    assert latents.shape == (1, 24, 1, 3, 3)
    with torch.inference_mode(), set_forward_context(0, None):
        torch.testing.assert_close(
            tiny_image.decode_base(latents, frame_num=1, process_image=True),
            tiny_image.tiled_decode(latents)[:, :, 0],
            rtol=0,
            atol=0,
        )
        assert tiny_image.decode_image_raw(latents).shape == (1, 3, 48, 48)


def test_posterior_sample_without_generator_uses_global_cpu_rng():
    posterior = DiagonalGaussianDistribution(torch.randn(1, 48, 1, 2, 2))
    with torch.random.fork_rng():
        torch.manual_seed(812)
        noise = torch.randn(posterior.mean.shape)
        expected = noise.mul_(posterior.std).add_(posterior.mean)
        torch.manual_seed(812)
        actual = posterior.sample()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_attention_and_tile_batching_share_decode_gate(tiny_image):
    gate = tiny_image.decode_fast_path_gate
    assert maybe_optimize_minimax_h3_vae(tiny_image) is tiny_image
    attention = [block.attn for block in tiny_image.decoder.transformer_blocks]
    assert all(layer._sgl_gate is gate for layer in attention)
    with use_vae_fast_path(tiny_image, True):
        assert gate.enabled
        with use_vae_fast_path(tiny_image, False):
            assert not any(layer._sgl_gate.enabled for layer in attention)
        assert all(layer._sgl_gate.enabled for layer in attention)
    assert not gate.enabled


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_tile_batching_bounds_order_and_exact_restoration(
    tiny_image, monkeypatch, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA tile batching requires a CUDA device")
    calls = []

    def decode(tiles):
        calls.append(tiles.shape[0])
        return tiles * 2

    monkeypatch.setattr(tiny_image, "decode", decode)
    tiles = [torch.full((2, 1, 1, 1), i, device=device) for i in range(130)]
    indices = list(range(129, -1, -1))
    expected = [tiles[i] * 2 for i in indices]
    with use_vae_fast_path(tiny_image, True):
        outputs = tiny_image._run_tile_tasks(tiles, indices, decode)
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert calls == ([128, 128, 4] if device == "cuda" else [2] * 130)
    calls.clear()
    tiny_image._run_tile_tasks(tiles, indices, decode)
    assert calls == [2] * 130

    # Encoder execution must never inherit decode batching.
    calls.clear()
    with use_vae_fast_path(tiny_image, True):
        tiny_image._run_tile_tasks(tiles, indices, lambda x: decode(x))
    assert calls == [2] * 130


def _decode_tiles_distributed(rank, rendezvous):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline_config import (
        normalized_args,
    )

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    set_global_server_args(normalized_args())
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=2,
        ulysses_degree=2,
        distributed_init_method=rendezvous,
        dist_timeout=60,
    )
    try:
        vae = _tiny_image()
        calls = []

        def decode(tiles):
            calls.append(tiles.shape[0])
            return tiles * 2

        vae.decode = decode
        for count in (1, 5, 129):
            tiles = [
                torch.full((2, 1, 1, 1), i, device=f"cuda:{rank}") for i in range(count)
            ]
            indices = list(range(count))
            with use_vae_fast_path(vae, True):
                outputs = vae._run_tile_tasks(tiles, indices, decode)
            for actual, tile in zip(outputs, tiles, strict=True):
                torch.testing.assert_close(actual, tile * 2, rtol=0, atol=0)
            local_samples = 2 * len(range(rank, count, 2)) if count > 1 else 2
            assert sum(calls) == local_samples
            assert max(calls) <= 128
            calls.clear()

            vae.sglang_config.use_parallel_tiling = False
            with use_vae_fast_path(vae, True):
                vae._run_tile_tasks(tiles, indices, decode)
            assert sum(calls) == 2 * count
            vae.sglang_config.use_parallel_tiling = True
            calls.clear()

            vae._run_tile_tasks(tiles, indices, decode)
            assert calls == [2] * count
            calls.clear()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_lossless_decode_distributes_complete_tiles(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("Tile decode needs two CUDA devices")
    mp.spawn(
        _decode_tiles_distributed,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
    )
