# SPDX-License-Identifier: Apache-2.0
"""Native KL VAE variants retain their interfaces and Diffusers tiling numerics."""

import pytest
import torch
from diffusers import AutoencoderKL as ReferenceAutoencoderKL
from diffusers.models.attention_processor import AttnAddedKVProcessor, AttnProcessor

from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.models.vaes.stable_diffusion import (
    StableDiffusionVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
)


@pytest.mark.parametrize("flux2", [False, True])
@pytest.mark.parametrize("quant_conv", [False, True])
@pytest.mark.parametrize("mode", ["direct", "sliced", "tiled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_kl_encode_decode_matches_diffusers(flux2, quant_conv, mode, dtype):
    arch = dict(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(8, 16),
        layers_per_block=1,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
        use_quant_conv=quant_conv,
        use_post_quant_conv=quant_conv,
        mid_block_add_attention=True,
    )
    config = Flux2VAEConfig() if flux2 else StableDiffusionVAEConfig()
    config.update_model_arch(arch)
    if flux2:
        config.update_model_arch(
            dict(batch_norm_eps=1e-4, batch_norm_momentum=0.1, patch_size=(2, 2))
        )
    config.use_parallel_decode = False
    model = (AutoencoderKLFlux2 if flux2 else AutoencoderKL)(config).eval()
    reference = ReferenceAutoencoderKL(**arch).eval()
    reference.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("bn.")},
        strict=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device, dtype=dtype)
    reference.to(device=device, dtype=dtype)
    model.use_slicing = reference.use_slicing = mode == "sliced"
    model.use_tiling = reference.use_tiling = mode == "tiled"
    x = torch.randn(2, 3, 24, 28, device=device, dtype=dtype)
    expected = reference.encode(x).latent_dist
    processors = model.attn_processors
    processor_keys = tuple(processors)
    model.set_attn_processor(processors)
    assert processors == {}
    assert tuple(model.attn_processors) == processor_keys
    with pytest.raises(ValueError, match="number of processors"):
        model.set_attn_processor({})
    for _ in range(2):
        posterior = model.encode(x.unsqueeze(2) if flux2 else x)
        if not flux2:
            posterior = posterior.latent_dist
        torch.testing.assert_close(
            posterior.parameters, expected.parameters, rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.decode(posterior.mode()),
            reference.decode(expected.mode()).sample,
            rtol=0,
            atol=0,
        )
    # direct tiled entrypoints retain both the structured and tuple contracts
    if mode == "tiled":
        torch.testing.assert_close(
            model.tiled_encode(x, return_dict=False)[0].parameters,
            expected.parameters,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.tiled_decode(posterior.mode(), return_dict=False)[0],
            reference.tiled_decode(expected.mode()).sample,
            rtol=0,
            atol=0,
        )
    for processor_cls in (AttnProcessor, AttnAddedKVProcessor):
        model.set_attn_processor(processor_cls())
        model.set_default_attn_processor()
        assert all(type(p) is processor_cls for p in model.attn_processors.values())
    model.set_attn_processor(object())
    with pytest.raises(ValueError, match="Cannot call"):
        model.set_default_attn_processor()
