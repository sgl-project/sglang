# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import sys
import types

import torch

if "comfy" not in sys.modules:
    # The plugin only ever runs inside ComfyUI; stub the bit of
    # comfy.ldm.common_dit the Qwen-Image adapter imports at module scope so
    # this contract is testable without a ComfyUI install or a GPU. Mirrors
    # the stubbing convention in
    # apps/ComfyUI_SGLDiffusion/test/test_h3_request.py.
    comfy = types.ModuleType("comfy")
    comfy_ldm = types.ModuleType("comfy.ldm")
    comfy_ldm_common_dit = types.ModuleType("comfy.ldm.common_dit")

    def _pad_to_patch_size(x, patch_size):
        return x

    comfy_ldm_common_dit.pad_to_patch_size = _pad_to_patch_size
    comfy_ldm.common_dit = comfy_ldm_common_dit
    comfy.ldm = comfy_ldm

    sys.modules["comfy"] = comfy
    sys.modules["comfy.ldm"] = comfy_ldm
    sys.modules["comfy.ldm.common_dit"] = comfy_ldm_common_dit

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image import (
    QwenImageEditAdapter,
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


def test_qwen_image_edit_pack_keeps_all_ref_latents() -> None:
    adapter = QwenImageEditAdapter()
    x = torch.ones(1, 16, 1, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)
    ref_a = torch.ones(1, 16, 1, 64, 64)
    ref_b = torch.ones(1, 16, 1, 32, 96)
    ref_c = torch.ones(1, 16, 1, 48, 48)

    packed = adapter.pack(
        x, timestep, context, ref_latents=[ref_a, ref_b, ref_c]
    )

    sizes = packed.extra_req["vae_image_sizes"]
    assert sizes == [(64, 64), (96, 32), (48, 48)]

    expected_tokens = sum(
        (h // 2) * (w // 2) for h, w in [(64, 64), (32, 96), (48, 48)]
    )
    assert packed.extra_req["image_latent"].shape[1] == expected_tokens
