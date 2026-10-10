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


def test_flux_pack_zero_fills_missing_pooled() -> None:
    from sglang.multimodal_gen.runtime.layers.visual_embedding import (
        CombinedTimestepGuidanceTextProjEmbeddings,
    )

    # ComfyUI calls Flux with y=None when the cond has no pooled_output; its
    # native model zero-fills y, so the adapter must too.
    adapter = FluxAdapter()
    x = torch.zeros(2, 16, 8, 8)
    context = torch.ones(2, 8, 4096, dtype=torch.bfloat16)
    packed = adapter.pack(x, torch.tensor([0.5, 0.5]), context, y=None)
    y = packed.pooled_embeds[0]
    assert y.shape == (2, 768)
    assert y.dtype == context.dtype
    assert not y.any()
    assert packed.prompt_embeds[0] is y

    # The JIT timestep_embedding kernel only accepts CUDA tensors, so run the
    # embedding where the kernel can run.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = CombinedTimestepGuidanceTextProjEmbeddings(
        embedding_dim=32, pooled_projection_dim=768
    ).to(device)
    out = embed(
        torch.tensor([500.0, 500.0], device=device),
        torch.tensor([3.5, 3.5], device=device),
        y.float().to(device),
    )
    assert out.shape == (2, 32)


def test_flux_pack_rejects_kontext_reference_latents() -> None:
    """ComfyUI's Flux forward passes ref_latents from the ReferenceLatent node;
    FluxAdapter must reject it instead of silently running plain T2I."""
    adapter = FluxAdapter()
    x = torch.ones(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    with pytest.raises(ValueError, match="Kontext"):
        adapter.pack(
            x,
            timestep,
            context,
            y=torch.ones(1, 768),
            ref_latents=[torch.ones(1, 16, 8, 8)],
        )


def test_flux_pack_rejects_controlnet_control() -> None:
    """ComfyUI's Flux forward passes control from a ControlNet node; FluxAdapter
    must reject it instead of silently running unconditioned."""
    adapter = FluxAdapter()
    x = torch.ones(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    with pytest.raises(ValueError, match="ControlNet"):
        adapter.pack(x, timestep, context, y=torch.ones(1, 768), control={"input": []})


def test_flux_pack_and_unpack_roundtrip_odd_latent_size() -> None:
    """Regression: a width/height not divisible by patch_size=2 (e.g. a
    1032px-wide ComfyUI latent, 1032 // 8 = 129) raised a view() RuntimeError
    in _pack_latents before it was padded like QwenImageAdapter."""
    adapter = FluxAdapter()
    x = torch.arange(1 * 16 * 8 * 9, dtype=torch.float32).reshape(1, 16, 8, 9)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    y = torch.ones(1, 768)
    packed = adapter.pack(x, timestep, context, y=y, guidance=torch.tensor([1.0]))
    out = adapter.unpack(packed.latents, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)
