# SPDX-License-Identifier: Apache-2.0
"""Kohya/musubi `lora_unet_*` LoRAs for Qwen-Image and Z-Image normalize to native names."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.lora import format_adapter
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)

RANK, ALPHA, DIM = 4, 8.0, 16


def _kohya(modules: list[str]) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    sd = {}
    for m in modules:
        sd[f"lora_unet_{m}.lora_down.weight"] = torch.randn(RANK, DIM)
        sd[f"lora_unet_{m}.lora_up.weight"] = torch.randn(DIM, RANK)
        sd[f"lora_unet_{m}.alpha"] = torch.tensor(ALPHA)
    return sd


@pytest.mark.parametrize(
    "fmt,modules,expected",
    [
        (
            LoRAFormat.KOHYA_Z_IMAGE,
            [
                "layers_0_attention_to_q",
                "layers_0_attention_to_out_0",
                "layers_0_feed_forward_w1",
                "context_refiner_1_attention_to_k",
            ],
            [
                "layers.0.attention.to_q",
                "layers.0.attention.to_out.0",
                "layers.0.feed_forward.w1",
                "context_refiner.1.attention.to_k",
            ],
        ),
        (
            LoRAFormat.KOHYA_QWEN_IMAGE,
            [
                "transformer_blocks_0_attn_to_q",
                "transformer_blocks_0_attn_add_q_proj",
                "transformer_blocks_0_img_mlp_net_0_proj",
                "transformer_blocks_0_txt_mod_1",
            ],
            [
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.add_q_proj",
                "transformer_blocks.0.img_mlp.net.0.proj",
                "transformer_blocks.0.txt_mod.1",
            ],
        ),
    ],
)
def test_kohya_dit_lora_normalizes_to_native_names(fmt, modules, expected):
    sd = _kohya(modules)
    assert detect_lora_format_from_state_dict(sd) == fmt

    out = normalize_lora_state_dict(sd)

    assert set(out) == {
        f"{n}.{part}"
        for n in expected
        for part in ("lora_A.weight", "lora_B.weight", "alpha")
    }
    # alpha / rank is folded into A/B, and alpha = rank keeps the loader from
    # scaling again.
    for m, n in zip(modules, expected):
        ref = (ALPHA / RANK) * (
            sd[f"lora_unet_{m}.lora_up.weight"] @ sd[f"lora_unet_{m}.lora_down.weight"]
        )
        got = out[f"{n}.lora_B.weight"] @ out[f"{n}.lora_A.weight"]
        torch.testing.assert_close(got, ref)


def test_kohya_mmdit_without_qwen_modules_is_not_qwen_image():
    # Kohya export of a FLUX-style MMDiT: diffusers `transformer_blocks` attention
    # names plus `single_transformer_blocks`, no Qwen img/txt mlp or mod.
    sd = _kohya(
        [
            "transformer_blocks_0_attn_to_q",
            "transformer_blocks_0_attn_add_q_proj",
            "single_transformer_blocks_0_attn_to_q",
        ]
    )
    assert detect_lora_format_from_state_dict(sd) == LoRAFormat.NON_DIFFUSERS_SD


@pytest.mark.parametrize(
    "dit_class_name,expected",
    [
        (None, LoRAFormat.NON_DIFFUSERS_SD),
        ("FluxTransformer2DModel", LoRAFormat.NON_DIFFUSERS_SD),
        ("QwenImageTransformer2DModel", LoRAFormat.KOHYA_QWEN_IMAGE),
        ("FSDPQwenImageTransformer2DModel", LoRAFormat.KOHYA_QWEN_IMAGE),
    ],
)
def test_attention_only_qwen_kohya_resolves_by_loaded_dit(dit_class_name, expected):
    # Trained only on to_q / to_v: no img/txt mlp or mod names to identify Qwen-Image.
    modules = ["transformer_blocks_0_attn_to_q", "transformer_blocks_0_attn_to_v"]
    sd = _kohya(modules)
    assert detect_lora_format_from_state_dict(sd, dit_class_name) == expected

    if expected == LoRAFormat.KOHYA_QWEN_IMAGE:
        out = normalize_lora_state_dict(sd, dit_class_name=dit_class_name)
        assert set(out) == {
            f"transformer_blocks.0.attn.to_{p}.{part}"
            for p in "qv"
            for part in ("lora_A.weight", "lora_B.weight", "alpha")
        }


def test_qwen_image_21_dit_does_not_take_qwen_image_kohya_path():
    # Qwen-Image 2.1 names its MLP gate differently; the Qwen-Image converter
    # would only half-map such a file.
    sd = _kohya(["transformer_blocks_0_img_mlp_net_0_proj"])
    assert (
        detect_lora_format_from_state_dict(sd, "QwenImage21Transformer2DModel")
        == LoRAFormat.NON_DIFFUSERS_SD
    )


@pytest.mark.parametrize("fmt", [LoRAFormat.KOHYA_QWEN_IMAGE, LoRAFormat.KOHYA_Z_IMAGE])
def test_alpha_fallback_for_modules_without_their_own_alpha(fmt):
    # Modules without `.alpha` take the caller's override, else the file's
    # global alpha, else alpha = rank; it is folded once, in the conversion.
    module, native = {
        LoRAFormat.KOHYA_QWEN_IMAGE: (
            "transformer_blocks_0_img_mlp_net_0_proj",
            "transformer_blocks.0.img_mlp.net.0.proj",
        ),
        LoRAFormat.KOHYA_Z_IMAGE: (
            "layers_0_attention_to_q",
            "layers.0.attention.to_q",
        ),
    }[fmt]
    sd = {k: v for k, v in _kohya([module]).items() if not k.endswith(".alpha")}
    up_down = (
        sd[f"lora_unet_{module}.lora_up.weight"]
        @ sd[f"lora_unet_{module}.lora_down.weight"]
    )
    cases = [
        ({"adapter_config": {"lora_alpha": 4}}, 4),
        ({"adapter_config": {"lora_alpha": 8}}, 8),
        ({"adapter_config": {"lora_alpha": 8}, "lora_alpha": 2}, 2),
        ({}, RANK),
    ]
    for kwargs, alpha in cases:
        out = normalize_lora_state_dict(sd, **kwargs)
        got = out[f"{native}.lora_B.weight"] @ out[f"{native}.lora_A.weight"]
        torch.testing.assert_close(got, (alpha / RANK) * up_down)
        assert out[f"{native}.alpha"].item() == RANK


def test_override_does_not_replace_a_module_alpha():
    # As for PEFT files, a module's own alpha wins over the caller's override.
    sd = _kohya(["layers_0_attention_to_q"])
    out = normalize_lora_state_dict(sd, lora_alpha=2)
    got = (
        out["layers.0.attention.to_q.lora_B.weight"]
        @ out["layers.0.attention.to_q.lora_A.weight"]
    )
    ref = (
        sd["lora_unet_layers_0_attention_to_q.lora_up.weight"]
        @ sd["lora_unet_layers_0_attention_to_q.lora_down.weight"]
    )
    torch.testing.assert_close(got, (ALPHA / RANK) * ref)


def test_z_image_adaln_modulation_keeps_its_module_name():
    out = normalize_lora_state_dict(
        _kohya(
            [
                "layers_0_attention_to_q",
                "layers_0_adaLN_modulation_0",
                "noise_refiner_1_adaLN_modulation_0",
            ]
        )
    )
    assert {k.rsplit(".", 2)[0] for k in out if k.endswith("lora_A.weight")} == {
        "layers.0.attention.to_q",
        "layers.0.adaLN_modulation.0",
        "noise_refiner.1.adaLN_modulation.0",
    }


def test_failed_kohya_conversion_raises_instead_of_loading_nothing(monkeypatch):
    def broken(state_dict):
        raise KeyError("alpha")

    monkeypatch.setattr(
        format_adapter.lcu, "_convert_non_diffusers_z_image_lora_to_diffusers", broken
    )
    with pytest.raises(ValueError, match="Kohya Z-Image"):
        normalize_lora_state_dict(_kohya(["layers_0_attention_to_q"]))


def test_dot_format_z_image_lora_without_alpha_loads():
    # Some trainers write `to.q.lora.down.weight`; the converter looks their
    # alpha up under the `to_q` name.
    torch.manual_seed(0)
    down, up = torch.randn(RANK, DIM), torch.randn(DIM, RANK)
    out = normalize_lora_state_dict(
        {
            "lora_unet_layers_0_attention_to.q.lora.down.weight": down,
            "lora_unet_layers_0_attention_to.q.lora.up.weight": up,
        }
    )
    got = (
        out["layers.0.attention.to_q.lora_B.weight"]
        @ out["layers.0.attention.to_q.lora_A.weight"]
    )
    torch.testing.assert_close(got, up @ down)
