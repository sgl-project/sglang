"""Popular Hub LoRAs must reach every layer they train.

A LoRA whose keys the loader cannot map still loads: the unmatched layers keep
their base weights and generation runs without an error, so a format or naming
regression is silent. This test reads only safetensors headers (no weights) of
each base transformer and of widely used LoRAs, runs the loader's own key
normalization and name mapping, and checks that every LoRA module lands on a
weight of the base DiT.

When a fix makes a new LoRA format load, add its most downloaded files here.
"""

import fnmatch
from functools import lru_cache

import pytest
import torch
from huggingface_hub import HfApi

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    LTX23PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
    Wan2_2_TI2V_5B_Config,
    WanT2V720PConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
    ZImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    map_lora_state_dict_to_dit,
    strip_diffusers_transformer_prefix,
)

# Base model -> (its pipeline config as registered, transformer weight files).
# Gated repos are left out because CI has no Hub token.
BASES = {
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": (WanT2V720PConfig, "transformer/*"),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": (Wan2_2_TI2V_5B_Config, "transformer/*"),
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": (Wan2_2_T2V_A14B_Config, "transformer/*"),
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": (Wan2_2_I2V_A14B_Config, "transformer/*"),
    "Qwen/Qwen-Image": (QwenImagePipelineConfig, "transformer/*"),
    "Qwen/Qwen-Image-Edit-2511": (
        QwenImageEditPlus_2511_PipelineConfig,
        "transformer/*",
    ),
    "black-forest-labs/FLUX.2-klein-4B": (Flux2KleinPipelineConfig, "transformer/*"),
    "Tongyi-MAI/Z-Image-Turbo": (ZImagePipelineConfig, "transformer/*"),
    "Lightricks/LTX-2": (LTX2PipelineConfig, "ltx-2-19b-dev.safetensors"),
    "Lightricks/LTX-2.3": (LTX23PipelineConfig, "ltx-2.3-22b-dev.safetensors"),
}

# (base model, LoRA repo, LoRA file, layers it reaches): the most downloaded
# LoRAs per base model, across the PEFT/diffusers, Kohya, XLabs, ComfyUI and
# lightx2v layouts.
CASES = [
    (
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "AlekseyCalvin/Jodorowsky_Wan_14b_LoRA_BySilverAgePoets",
        "Jodorowsky_000001600.safetensors",
        400,
    ),
    (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets",
        "HSTcolorWan225b_000000200.safetensors",
        300,
    ),
    (
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "lightx2v/Wan2.2-Distill-Loras",
        "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        400,
    ),
    (
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "renderartist/technically-color-wan",
        "Technically_Color_WAN_2_2_T2V_v1_000003750_high_noise.safetensors",
        400,
    ),
    (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "ostris/wan22_i2v_14b_orbit_shot_lora",
        "wan22_14b_i2v_orbit_high_noise.safetensors",
        400,
    ),
    (
        "Qwen/Qwen-Image",
        "lightx2v/Qwen-Image-2512-Lightning",
        "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        600,
    ),
    (
        "Qwen/Qwen-Image-Edit-2511",
        "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        600,
    ),
    (
        "Qwen/Qwen-Image-Edit-2511",
        "lrzjason/Anything2Real",
        "anything2real_2601.safetensors",
        480,
    ),
    (
        "black-forest-labs/FLUX.2-klein-4B",
        "Limbicnation/pixel-art-lora",
        "pytorch_lora_weights.comfyui.safetensors",
        86,
    ),
    (
        "black-forest-labs/FLUX.2-klein-4B",
        "radames/FLUX.2-klein-Sana-Sprint",
        "pytorch_lora_weights.safetensors",
        100,
    ),
    (
        "Tongyi-MAI/Z-Image-Turbo",
        "ostris/zimage_turbo_training_adapter",
        "zimage_turbo_training_adapter_v2.safetensors",
        150,
    ),
    (
        "Tongyi-MAI/Z-Image-Turbo",
        "tarn59/pixel_art_style_lora_z_image_turbo",
        "pixel_art_style_z_image_turbo.safetensors",
        150,
    ),
    (
        "Lightricks/LTX-2",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        480,
    ),
    (
        "Lightricks/LTX-2.3",
        "joyfox/LTX-2.3-Transition-LORA",
        "ltx2.3-transition.safetensors",
        576,
    ),
]

# Tensors up to this size are built as zeros (format converters read alphas
# and small LoRA factors); larger ones are meta tensors, which keep the shape.
_MAX_MATERIALIZED_NUMEL = 1 << 21


def _header_state_dict(repo_id: str, filename: str) -> dict[str, torch.Tensor]:
    tensors = HfApi().parse_safetensors_file_metadata(repo_id, filename).tensors
    state_dict = {}
    for name, info in tensors.items():
        shape = list(info.shape)
        numel = 1
        for size in shape:
            numel *= size
        state_dict[name] = (
            torch.zeros(shape)
            if numel <= _MAX_MATERIALIZED_NUMEL
            else torch.empty(shape, device="meta")
        )
    return state_dict


@lru_cache(maxsize=None)
def _arch_config(base_model: str):
    pipeline_config_cls, _ = BASES[base_model]
    return pipeline_config_cls().dit_config.arch_config


@lru_cache(maxsize=None)
def _base_layers(base_model: str) -> frozenset[str]:
    """Names of the DiT layers a LoRA can target, from the base checkpoint."""
    arch = _arch_config(base_model)
    param_names_mapping_fn = get_param_names_mapping(arch.param_names_mapping)
    api = HfApi()
    _, pattern = BASES[base_model]
    files = [
        name
        for name in fnmatch.filter(api.list_repo_files(base_model), pattern)
        if name.endswith(".safetensors")
    ]
    assert files, f"no transformer weights in {base_model}"
    layers = set()
    for filename in files:
        tensors = api.parse_safetensors_file_metadata(base_model, filename).tensors
        for name, info in tensors.items():
            if not name.endswith(".weight") or len(info.shape) != 2:
                continue
            target_name, _, _ = param_names_mapping_fn(
                name.replace("model.diffusion_model.", "")
            )
            layer = target_name.removesuffix(".weight")
            if not any(excluded in layer for excluded in arch.exclude_lora_layers):
                layers.add(layer)
    return frozenset(layers)


def _lora_layers(base_model: str, lora_repo: str, lora_file: str) -> set[str]:
    """Layers the loader would apply this LoRA to (both A and B present)."""
    arch = _arch_config(base_model)
    lora_state_dict = strip_diffusers_transformer_prefix(
        normalize_lora_state_dict(
            _header_state_dict(lora_repo, lora_file), adapter_config={}
        )
    )
    adapter = map_lora_state_dict_to_dit(
        lora_state_dict,
        get_param_names_mapping(arch.param_names_mapping),
        get_param_names_mapping(arch.lora_param_names_mapping),
        lora_alpha=None,
        device="meta",
    )
    return {
        name.removesuffix(".lora_A")
        for name in adapter
        if name.endswith(".lora_A")
        and name.removesuffix(".lora_A") + ".lora_B" in adapter
    }


@pytest.mark.parametrize(
    "base_model, lora_repo, lora_file, expected_layers",
    CASES,
    ids=[f"{case[1]}/{case[2]}" for case in CASES],
)
def test_hub_lora_reaches_its_layers(base_model, lora_repo, lora_file, expected_layers):
    layers = _lora_layers(base_model, lora_repo, lora_file)
    unmatched = sorted(layers - _base_layers(base_model))
    assert layers, f"{lora_repo}/{lora_file} maps to no layer of {base_model}"
    assert not unmatched, (
        f"{len(unmatched)}/{len(layers)} LoRA layers of {lora_repo}/{lora_file} "
        f"are not layers of {base_model}, e.g. {unmatched[:5]}"
    )
    if expected_layers is not None:
        assert len(layers) == expected_layers
