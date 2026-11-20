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


def test_qwen_image_21_kohya_lora_is_rejected_instead_of_loading_nothing():
    # No Kohya name maps to a Qwen-Image 2.1 layer, so the generic fallback
    # would report success with zero layers applied.
    sd = _kohya(["transformer_blocks_0_img_mlp_out", "transformer_blocks_0_attn_to_q"])
    for dit_class_name in (
        "QwenImage21Transformer2DModel",
        "FSDPQwenImage21Transformer2DModel",
    ):
        with pytest.raises(ValueError, match="Qwen-Image 2.1"):
            normalize_lora_state_dict(sd, dit_class_name=dit_class_name)


_ALPHA_PATTERN_MODULES = {
    LoRAFormat.KOHYA_QWEN_IMAGE: (
        "transformer_blocks_0_attn_to_{}",
        "transformer_blocks.0.attn.to_{}",
        "QwenImageTransformer2DModel",
    ),
    LoRAFormat.KOHYA_Z_IMAGE: (
        "layers_0_attention_to_{}",
        "layers.0.attention.to_{}",
        "ZImageTransformer2DModel",
    ),
}


@pytest.mark.parametrize("fmt", list(_ALPHA_PATTERN_MODULES))
def test_alpha_pattern_is_checked_against_the_original_alpha(fmt):
    kohya, native, dit_class_name = _ALPHA_PATTERN_MODULES[fmt]
    sd = _kohya([kohya.format("q"), kohya.format("v")])
    # to_q carries no alpha of its own, so its pattern value is used.
    del sd[f"lora_unet_{kohya.format('q')}.alpha"]
    out = normalize_lora_state_dict(
        sd,
        adapter_config={"lora_alpha": 1, "alpha_pattern": {"to_v": 8, "to_q": 2}},
        dit_class_name=dit_class_name,
    )
    for proj, alpha in (("v", ALPHA), ("q", 2)):
        k, n = kohya.format(proj), native.format(proj)
        got = out[f"{n}.lora_B.weight"] @ out[f"{n}.lora_A.weight"]
        ref = (
            sd[f"lora_unet_{k}.lora_up.weight"] @ sd[f"lora_unet_{k}.lora_down.weight"]
        )
        torch.testing.assert_close(got, (alpha / RANK) * ref)
        assert out[f"{n}.alpha"].item() == RANK

    with pytest.raises(ValueError, match="conflict with its alpha 8"):
        normalize_lora_state_dict(
            sd,
            adapter_config={"alpha_pattern": {"to_v": 2}},
            dit_class_name=dit_class_name,
        )


def test_alpha_pattern_on_fused_z_image_qkv_must_agree_across_q_k_v():
    # Newer diffusers split a fused `qkv` into to_q/k/v that share one alpha.
    torch.manual_seed(0)
    down, up = torch.randn(RANK, DIM), torch.randn(3 * DIM, RANK)
    sd = {
        "lora_unet_layers_0_attention_qkv.lora.down.weight": down,
        "lora_unet_layers_0_attention_qkv.lora.up.weight": up,
    }

    def load(adapter_config, extra=None):
        return normalize_lora_state_dict(
            {**sd, **(extra or {})},
            adapter_config=adapter_config,
            dit_class_name="ZImageTransformer2DModel",
        )

    def assert_scale(out, alpha):
        for proj, chunk in zip("qkv", up.chunk(3)):
            n = f"layers.0.attention.to_{proj}"
            got = out[f"{n}.lora_B.weight"] @ out[f"{n}.lora_A.weight"]
            torch.testing.assert_close(got, (alpha / RANK) * (chunk @ down))

    try:
        out = load({"alpha_pattern": {"to_q": 2, "to_k": 2, "to_v": 2}})
    except ValueError as error:
        # Releases without the split (e.g. the pinned 0.37) cannot convert it.
        assert "Could not convert" in str(error)
        return
    assert_scale(out, 2)
    # A pattern on some of q/k/v is fine when it equals the alpha the rest use.
    own_alpha = {"lora_unet_layers_0_attention_qkv.alpha": torch.tensor(8.0)}
    assert_scale(load({"alpha_pattern": {"to_q": 8}}, own_alpha), 8)
    assert_scale(load({"lora_alpha": 8, "alpha_pattern": {"to_q": 8}}), 8)
    with pytest.raises(ValueError, match="conflict with each other"):
        load({"alpha_pattern": {"to_k": 8}})
    with pytest.raises(ValueError, match="conflict with its alpha 8"):
        load({"alpha_pattern": {"to_q": 8, "to_k": 2}}, own_alpha)


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
