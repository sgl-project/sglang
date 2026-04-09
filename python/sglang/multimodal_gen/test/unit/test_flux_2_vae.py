import importlib.util

import pytest

if importlib.util.find_spec("triton.compiler") is None:
    pytest.skip(
        "triton.compiler is required to import sglang.multimodal_gen",
        allow_module_level=True,
    )

from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
)


def test_decoder_block_out_channels_override():
    config = Flux2VAEConfig()
    config.update_model_arch(
        {
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [96, 192, 384, 384],
            "latent_channels": 32,
            "patch_size": [2, 2],
        }
    )
    config.post_init()

    vae = AutoencoderKLFlux2(config)

    assert tuple(vae.encoder.conv_in.weight.shape) == (128, 3, 3, 3)
    assert tuple(vae.decoder.conv_in.weight.shape) == (384, 32, 3, 3)
    assert tuple(vae.decoder.up_blocks[0].resnets[0].conv1.weight.shape) == (
        384,
        384,
        3,
        3,
    )
