# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for single-GPU Qwen-Image-2.1 VAE tiling."""

from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import autoencoder_kl_qwenimage21 as q


def tiny_vae(**arch_overrides):
    torch.manual_seed(42)
    arch = dict(
        in_channels=3,
        base_dim=2,
        decoder_base_dim=2,
        z_dim=2,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        is_residual=True,
    )
    arch.update(arch_overrides)
    model = (
        q.AutoencoderKLQwenImage21(
            QwenImage21VAEConfig(
                arch_config=QwenImage21VAEArchConfig(**arch),
                use_tiling=True,
                use_temporal_tiling=False,
                use_parallel_tiling=False,
            )
        )
        .double()
        .eval()
    )
    ratio = model.spatial_compression_ratio
    model.tile_sample_min_height = 24 * ratio
    model.tile_sample_min_width = 32 * ratio
    model.tile_sample_stride_height = 16 * ratio
    model.tile_sample_stride_width = 19 * ratio
    return model


@pytest.mark.parametrize("encode", [True, False], ids=["encoder", "decoder"])
def test_context_tiles_match_untiled_output(encode):
    model = tiny_vae()
    # Neither axis ends on a tile boundary; compare the complete output.
    shape = (1, 3, 1, 512, 608) if encode else (1, 2, 1, 23, 26)
    source = torch.randn(shape, dtype=torch.float64)
    with torch.inference_mode():
        expected = model._encode(source) if encode else model._decode(source)
        actual = model.encode(source).parameters if encode else model.decode(source)
    torch.testing.assert_close(actual, expected, rtol=5e-9, atol=2e-10)


@pytest.mark.parametrize("world_size", [1, 2])
def test_context_tiling_is_single_process_only(monkeypatch, world_size):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    model = tiny_vae()
    marker = torch.tensor([0.375])
    context = Mock(return_value=marker)
    monkeypatch.setattr(model, "_context_tiled", context)
    for encode in (True, False):
        name = "spatial_tiled_encode" if encode else "spatial_tiled_decode"
        native = Mock(return_value=marker)
        monkeypatch.setattr(q.ParallelTiledVAE, name, native)
        shape = (1, 3, 1, 64, 80) if encode else (1, 2, 1, 3, 4)
        torch.testing.assert_close(getattr(model, name)(torch.zeros(shape)), marker)
        assert native.call_count == int(world_size > 1)
    assert context.call_count == (2 if world_size == 1 else 0)


@pytest.mark.parametrize("encode", [True, False], ids=["encoder", "decoder"])
def test_nonresidual_architecture_keeps_native_tiling(encode):
    model = tiny_vae(is_residual=False)
    shape = (1, 3, 1, 64, 80) if encode else (1, 2, 1, 3, 4)
    source = torch.randn(shape, dtype=torch.float64)
    name = "spatial_tiled_encode" if encode else "spatial_tiled_decode"
    with torch.inference_mode():
        expected = getattr(q.ParallelTiledVAE, name)(model, source)
        actual = getattr(model, name)(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiled_decode_preserves_output_range():
    model = tiny_vae()
    with torch.inference_mode():
        model.decoder.conv_out.weight.zero_()
        model.decoder.conv_out.bias.fill_(2.0)
        actual = model.decode(torch.zeros((1, 2, 1, 25, 34), dtype=torch.float64))
    assert torch.equal(actual, torch.ones_like(actual))
