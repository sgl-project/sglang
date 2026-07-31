# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.bagel import (
    BagelVAEArchConfig,
    BagelVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.bagel_vae import (
    BagelVAE,
    _AttnBlock,
)


@pytest.fixture
def tiny_vae() -> BagelVAE:
    arch = BagelVAEArchConfig(
        resolution=8,
        ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        z_channels=4,
        spatial_compression_ratio=1,
    )
    torch.manual_seed(0)
    return BagelVAE(BagelVAEConfig(arch_config=arch)).eval()


@pytest.fixture
def tiny_full_vae() -> BagelVAE:
    arch = BagelVAEArchConfig(
        resolution=8,
        ch=32,
        ch_mult=(1,),
        num_res_blocks=1,
        z_channels=4,
        spatial_compression_ratio=1,
    )
    torch.manual_seed(0)
    return BagelVAE(
        BagelVAEConfig(
            arch_config=arch,
            load_encoder=True,
            load_decoder=True,
        )
    ).eval()


def test_attention_uses_nchw_to_sequence_permutation() -> None:
    torch.manual_seed(1)
    block = _AttnBlock(32).eval()
    hidden_states = torch.arange(1 * 32 * 2 * 3, dtype=torch.float32).reshape(
        1, 32, 2, 3
    )
    normalized = block.norm(hidden_states)
    batch, channels, height, width = normalized.shape
    query = (
        block.q(normalized)
        .permute(0, 2, 3, 1)
        .reshape(batch, 1, height * width, channels)
    )
    key = (
        block.k(normalized)
        .permute(0, 2, 3, 1)
        .reshape(batch, 1, height * width, channels)
    )
    value = (
        block.v(normalized)
        .permute(0, 2, 3, 1)
        .reshape(batch, 1, height * width, channels)
    )
    expected = F.scaled_dot_product_attention(query, key, value)
    expected = (
        expected.reshape(batch, height, width, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    torch.testing.assert_close(block.attention(hidden_states), expected)


def test_decode_is_raw_and_does_not_apply_scale_shift(tiny_vae: BagelVAE) -> None:
    class _IdentityDecoder(nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents

    tiny_vae.decoder = _IdentityDecoder()
    latents = torch.randn(1, 4, 2, 3)
    torch.testing.assert_close(tiny_vae.decode(latents), latents)


def test_decoder_only_lifecycle_defaults() -> None:
    config = BagelVAEConfig()
    assert config.load_encoder is False
    assert config.load_decoder is True
    assert config.use_tiling is False
    assert config.use_parallel_decode is False
    assert config.get_vae_scale_factor() == 8
    assert config.arch_config.scaling_factor == pytest.approx(0.3611)
    assert config.arch_config.shift_factor == pytest.approx(0.1159)


def test_decoder_weight_loader_is_streaming_and_strict(tiny_vae: BagelVAE) -> None:
    expected = {
        name: torch.full_like(parameter, index + 1)
        for index, (name, parameter) in enumerate(tiny_vae.named_parameters())
    }
    weights = list(expected.items()) + [("encoder.ignored.weight", torch.ones(1))]

    loaded = tiny_vae.load_weights(iter(weights))

    assert loaded == set(expected)
    for name, parameter in tiny_vae.named_parameters():
        torch.testing.assert_close(parameter, expected[name])

    with pytest.raises(ValueError, match="unexpected VAE weights"):
        tiny_vae.load_weights([("unknown.weight", torch.ones(1))])


def test_decode_validates_nchw_rank(tiny_vae: BagelVAE) -> None:
    with pytest.raises(ValueError, match="expects NCHW"):
        tiny_vae.decode(torch.zeros(4, 2, 3))


def test_encode_uses_request_generator_without_global_rng(
    tiny_full_vae: BagelVAE,
) -> None:
    images = torch.randn(1, 3, 4, 4)
    global_state = torch.random.get_rng_state()

    first = tiny_full_vae.encode(
        images, generator=torch.Generator("cpu").manual_seed(123)
    )
    second = tiny_full_vae.encode(
        images, generator=torch.Generator("cpu").manual_seed(123)
    )

    torch.testing.assert_close(first, second)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_encode_applies_posterior_and_scale_shift_once(
    tiny_full_vae: BagelVAE,
) -> None:
    class FixedEncoder(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = images.shape
            mean = images.new_full((batch, 4, height, width), 0.25)
            log_variance = images.new_full((batch, 4, height, width), -100.0)
            return torch.cat((mean, log_variance), dim=1)

    tiny_full_vae.encoder = FixedEncoder()
    images = torch.zeros(1, 3, 2, 2)
    output = tiny_full_vae.encode(
        images, generator=torch.Generator("cpu").manual_seed(0)
    )
    expected = torch.full_like(
        output,
        tiny_full_vae.scaling_factor * (0.25 - tiny_full_vae.shift_factor),
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-6)


def test_full_vae_loader_requires_encoder_and_decoder(
    tiny_full_vae: BagelVAE,
) -> None:
    expected = [
        (name, torch.zeros_like(parameter))
        for name, parameter in tiny_full_vae.named_parameters()
    ]
    loaded = tiny_full_vae.load_weights(iter(expected))
    assert loaded == {name for name, _ in expected}

    decoder_only = [item for item in expected if item[0].startswith("decoder.")]
    with pytest.raises(ValueError, match="missing VAE weights"):
        tiny_full_vae.load_weights(iter(decoder_only))


def test_meta_initialization_materializes_decoder_for_streaming_load(
    tiny_vae: BagelVAE,
) -> None:
    with torch.device("meta"):
        vae = BagelVAE(tiny_vae.config)

    weights = [
        (name, torch.zeros(tuple(parameter.shape), dtype=parameter.dtype))
        for name, parameter in vae.named_parameters()
    ]
    loaded = vae.load_weights(iter(weights))
    vae.to("cpu")

    assert loaded == {name for name, _ in weights}
    assert not any(parameter.is_meta for parameter in vae.parameters())
