# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
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


def test_flux_pack_rejects_kontext_reference_latents() -> None:
    """ComfyUI's Flux forward passes ref_latents from the ReferenceLatent node;
    FluxAdapter must reject it instead of silently running plain T2I."""
    adapter = FluxAdapter()
    x = torch.ones(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    ref_latents = [torch.ones(1, 16, 8, 8)]
    with pytest.raises(ValueError, match="Kontext"):
        adapter.pack(x, timestep, context, ref_latents=ref_latents)


def test_flux_pack_rejects_controlnet_control() -> None:
    """ComfyUI's Flux forward passes a control dict for ControlNet; FluxAdapter
    must reject it instead of silently running unconditioned T2I."""
    adapter = FluxAdapter()
    x = torch.ones(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    with pytest.raises(ValueError, match="ControlNet"):
        adapter.pack(x, timestep, context, control={"input": []})
