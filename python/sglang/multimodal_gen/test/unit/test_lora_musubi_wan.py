"""musubi-tuner Wan LoRAs (lora_unet_blocks_N_*) load like the official Wan layout."""

import logging

import pytest
import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_musubi_wan_lora_to_diffusers,
)

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)

LOG = logging.getLogger(__name__)
RANK, DIM = 4, 8
# musubi module -> (official Wan name, diffusers name)
MODULES = {
    "self_attn_q": ("self_attn.q", "attn1.to_q"),
    "self_attn_k": ("self_attn.k", "attn1.to_k"),
    "self_attn_v": ("self_attn.v", "attn1.to_v"),
    "self_attn_o": ("self_attn.o", "attn1.to_out.0"),
    "cross_attn_q": ("cross_attn.q", "attn2.to_q"),
    "cross_attn_k": ("cross_attn.k", "attn2.to_k"),
    "cross_attn_v": ("cross_attn.v", "attn2.to_v"),
    "cross_attn_o": ("cross_attn.o", "attn2.to_out.0"),
    "ffn_0": ("ffn.0", "ffn.net.0.proj"),
    "ffn_2": ("ffn.2", "ffn.net.2"),
}
I2V_MODULES = {
    "cross_attn_k_img": ("cross_attn.k_img", "attn2.add_k_proj"),
    "cross_attn_v_img": ("cross_attn.v_img", "attn2.add_v_proj"),
}


def _musubi(modules, blocks=2, alpha=2.0):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for i in range(blocks):
        for m in modules:
            key = f"lora_unet_blocks_{i}_{m}"
            sd[f"{key}.lora_down.weight"] = torch.randn(RANK, DIM, generator=g)
            sd[f"{key}.lora_up.weight"] = torch.randn(DIM, RANK, generator=g)
            sd[f"{key}.alpha"] = torch.tensor(alpha)
    return sd


def _delta(sd, base):
    a, b = sd[f"{base}.lora_A.weight"], sd[f"{base}.lora_B.weight"]
    alpha = sd.get(f"{base}.alpha")
    return (b @ a) * (1.0 if alpha is None else alpha.item() / a.shape[0])


@pytest.mark.parametrize("modules", [MODULES, MODULES | I2V_MODULES])
def test_musubi_wan_matches_the_diffusers_conversion(modules):
    sd = _musubi(modules)
    assert detect_lora_format_from_state_dict(sd) == LoRAFormat.MUSUBI_WAN
    out = normalize_lora_state_dict(dict(sd), LOG)
    ref = _convert_musubi_wan_lora_to_diffusers({k: v.clone() for k, v in sd.items()})
    for i in range(2):
        for official, diffusers_name in modules.values():
            torch.testing.assert_close(
                _delta(out, f"diffusion_model.blocks.{i}.{official}"),
                _delta(ref, f"transformer.blocks.{i}.{diffusers_name}"),
            )


def test_musubi_wan_normalizes_like_the_official_wan_layout():
    sd = _musubi(MODULES)
    official = {}
    for key, tensor in sd.items():
        stem, suffix = key.split(".", 1)
        block, module = stem.removeprefix("lora_unet_blocks_").split("_", 1)
        name = MODULES[module][0]
        official[f"diffusion_model.blocks.{block}.{name}.{suffix}"] = tensor
    assert detect_lora_format_from_state_dict(official) == LoRAFormat.WAN
    assert normalize_lora_state_dict(dict(sd), LOG).keys() == (
        normalize_lora_state_dict(official, LOG).keys()
    )


@pytest.mark.parametrize(
    "extra_key",
    [
        "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight",  # Kohya MiniMax-H3
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",  # Kohya FLUX
    ],
)
def test_other_kohya_layouts_are_not_musubi_wan(extra_key):
    sd = _musubi(MODULES) | {extra_key: torch.zeros(RANK, DIM)}
    assert detect_lora_format_from_state_dict(sd) != LoRAFormat.MUSUBI_WAN


def _loader_names(state_dict):
    """Adapter names as LoRAPipeline.load_lora_adapter maps them for Wan."""
    from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    config = WanVideoArchConfig()
    lora_map = get_param_names_mapping(config.lora_param_names_mapping)
    param_map = get_param_names_mapping(config.param_names_mapping)
    names = set()
    for key in state_dict:
        if ".lora_" not in key:
            continue
        name = key.replace("diffusion_model.", "").removesuffix(".weight")
        name, _, _ = lora_map(name)
        names.add(param_map(name)[0])
    return names


def test_musubi_wan_lands_on_the_same_layers_as_a_diffusers_wan_lora():
    out = normalize_lora_state_dict(_musubi(MODULES), LOG)
    diffusers_layout = {
        f"blocks.{i}.{name}.lora_{ab}.weight": torch.zeros(1)
        for i in range(2)
        for _, name in MODULES.values()
        for ab in "AB"
    }
    names = _loader_names(out)
    assert len(names) == 2 * len(MODULES) * 2
    assert names == _loader_names(diffusers_layout)
