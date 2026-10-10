# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux2 import (
    Flux2Adapter,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)


def test_registered_comfyui_model_types() -> None:
    types = registered_model_types()
    assert "flux" in types
    assert "lumina2" in types
    assert get_adapter_class("lumina2") is ZImageAdapter
    assert get_adapter_class("flux") is FluxAdapter
    assert get_adapter_class("lumina2").pipeline_class_name == "ZImagePipeline"


def test_zimage_pack_sets_seq_lens_and_time_dim() -> None:
    adapter = ZImageAdapter()
    x = torch.ones(1, 16, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)
    packed = adapter.pack(x, timestep, context)
    assert packed.latents.shape == (1, 16, 1, 90, 160)
    assert packed.prompt_embeds[0].shape == (19, 2560)
    assert packed.prompt_seq_lens == [[19]]
    assert packed.height == 720
    assert packed.width == 1280
    assert torch.equal(packed.timesteps, timestep * 1000.0)

    pred = torch.ones(1, 16, 1, 90, 160)
    out = adapter.unpack(pred, packed, x)
    assert out.shape == x.shape


def test_flux_pack_and_unpack_roundtrip() -> None:
    adapter = FluxAdapter()
    x = torch.arange(1 * 16 * 8 * 8, dtype=torch.float32).reshape(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    y = torch.ones(1, 768)
    packed = adapter.pack(x, timestep, context, y=y, guidance=torch.tensor([1.0]))
    assert packed.latents.ndim == 3
    assert packed.pooled_embeds[0] is y
    assert packed.guidance_scale == 1.0
    out = adapter.unpack(packed.latents, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)

    default = adapter.pack(x, timestep, context, y=y)
    assert default.guidance_scale == 3.5


@pytest.mark.parametrize(
    "context_in_dim, guidance_embed, pipeline",
    [
        (15360, True, "Flux2ComfyUIPipeline"),
        (7680, True, "Flux2KleinBaseComfyUIPipeline"),
        (12288, False, "Flux2KleinComfyUIPipeline"),
    ],
)
def test_flux2_detected_family_selects_pipeline(
    context_in_dim, guidance_embed, pipeline
) -> None:
    config = SimpleNamespace(
        unet_config={"context_in_dim": context_in_dim, "guidance_embed": guidance_embed}
    )
    assert get_adapter_class("flux2") is Flux2Adapter
    assert Flux2Adapter.pipeline_class_for(config) == pipeline


def test_flux2_pack_flattens_latents_and_keeps_every_row() -> None:
    """ComfyUI stacks cond and uncond (B=2); each row needs its own seq len."""
    adapter = Flux2Adapter()
    x = torch.randn(2, 128, 4, 6)
    packed = adapter.pack(
        x,
        torch.tensor([0.5, 0.5]),
        torch.randn(2, 512, 7680),
        guidance=torch.tensor([4.0]),
    )
    assert packed.latents.shape == (2, 24, 128)
    assert packed.timesteps.tolist() == [500.0]
    assert packed.prompt_seq_lens == [[512, 512]]
    assert (packed.height, packed.width) == (64, 96)
    assert packed.guidance_scale == 4.0
    assert "image_latent" not in packed.extra_req

    pred = packed.latents.clone()
    out = adapter.unpack(pred, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_flux2_reference_latents_are_appended_with_distinct_ids() -> None:
    adapter = Flux2Adapter()
    x = torch.randn(1, 128, 4, 4)
    refs = [torch.randn(1, 128, 2, 2), torch.randn(1, 128, 2, 3)]
    packed = adapter.pack(
        x, torch.tensor([0.5]), torch.randn(1, 512, 7680), ref_latents=refs
    )
    assert packed.extra_req["image_latent"].shape == (1, 4 + 6, 128)
    ids = packed.extra_req["condition_image_latent_ids"][0]
    # One time coordinate per reference image: 10 then 20.
    assert ids[:4, 0].unique().tolist() == [10] and ids[4:, 0].unique().tolist() == [20]

    adapter.drop_cached_fields(packed)
    assert not packed.prompt_embeds
    # Dropped together, so the worker cache restores them as a pair.
    assert "image_latent" not in packed.extra_req
    assert "condition_image_latent_ids" not in packed.extra_req


def test_flux2_rejects_attention_masks_it_would_silently_drop() -> None:
    with pytest.raises(NotImplementedError, match="attention masks"):
        Flux2Adapter().pack(
            torch.randn(1, 128, 4, 4),
            torch.tensor([0.5]),
            torch.randn(1, 512, 7680),
            attention_mask=torch.ones(1, 16),
        )


def test_embedded_guidance_comes_from_the_request_only_for_flux2() -> None:
    """FluxGuidance must reach FLUX.2; other models send a placeholder 1.0."""
    from sglang.multimodal_gen.configs.pipeline_configs.flux import (
        Flux2PipelineConfig,
        FluxPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    batch = SimpleNamespace(guidance_scale=7.0)
    flux2 = SimpleNamespace(comfyui_mode=True, pipeline_config=Flux2PipelineConfig())
    flux1 = SimpleNamespace(comfyui_mode=True, pipeline_config=FluxPipelineConfig())
    native = SimpleNamespace(comfyui_mode=False, pipeline_config=Flux2PipelineConfig())
    assert DenoisingStage._comfyui_request_guidance(batch, flux2) == 7.0
    assert DenoisingStage._comfyui_request_guidance(batch, flux1) is None
    assert DenoisingStage._comfyui_request_guidance(batch, native) is None
