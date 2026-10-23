# SPDX-License-Identifier: Apache-2.0
"""GLM-Image DiT forwards and decode stats stay off the host's critical path."""

from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import glm_image
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs

# the DiT's fused kernels need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_glm_dit():
    config = GlmImageDitConfig()
    # a 256-wide hidden state keeps CUDA's fused kernels on supported widths
    for key, value in dict(
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        condition_dim=8,
        prior_vq_quantizer_codebook_size=32,
        text_embed_dim=24,
        time_embed_dim=16,
    ).items():
        setattr(config.arch_config, key, value)
    model = glm_image.GlmImageTransformer2DModel(config, {})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model


@torch.no_grad()
def test_prior_token_drop_does_not_sync(single_process_model_parallel):
    """Dropping the unconditional branch's prior tokens is a select: a boolean-mask
    write runs a nonzero, which blocks the host on every forward."""
    model = _tiny_glm_dit().to(device=_DEVICE, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, 16, 4, 4, generator=generator),
        encoder_hidden_states=torch.randn(1, 3, 24, generator=generator),
        prior_token_id=torch.randint(0, 32, (1, 4), generator=generator),
        prior_token_drop=torch.tensor([[True, False, True, False]]),
        timestep=torch.tensor([500.0]),
        target_size=torch.tensor([[64.0, 64.0]]),
        crop_coords=torch.zeros(1, 2),
    )
    inputs = {
        key: value.to(_DEVICE, torch.bfloat16 if value.is_floating_point() else None)
        for key, value in inputs.items()
    }
    with set_forward_context(current_timestep=0, attn_metadata=None):
        # the first call verifies the fused kernels against eager, once
        eager = model(**inputs)
        with forbid_host_syncs():
            model(**inputs)
        # the graph path zeroes the same rows with masked_fill
        with patch.object(glm_image, "is_in_breakable_cuda_graph", lambda: True):
            graphed = model(**inputs)
    torch.testing.assert_close(eager, graphed, atol=0, rtol=0)


def test_decode_latent_stats_are_uploaded_once():
    # decode scales with the first request's device tensors: a per-request
    # upload would stall the host until the denoising loop drains
    config = GlmImagePipelineConfig()
    vae_config = config.vae_config
    vae_config.latents_mean = tuple(0.25 * i - 2.0 for i in range(16))
    vae_config.latents_std = tuple(0.5 + 0.125 * i for i in range(16))
    device = torch.device("cpu")
    first = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
    second = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
    mean = torch.tensor(vae_config.latents_mean).reshape(1, 16, 1, 1)
    std = torch.tensor(vae_config.latents_std).reshape(1, 16, 1, 1)
    torch.testing.assert_close(first[0], 1.0 / std.to(torch.bfloat16), atol=0, rtol=0)
    torch.testing.assert_close(first[1], mean.to(torch.bfloat16), atol=0, rtol=0)
    assert first[0] is second[0] and first[1] is second[1]
