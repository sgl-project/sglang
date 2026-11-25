# SPDX-License-Identifier: Apache-2.0
"""Kohya/musubi `lora_unet_*` LoRAs for Qwen-Image and Z-Image normalize to native names."""

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.lora import format_adapter
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.peft_adapter import (
    load_peft_config,
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


@pytest.mark.parametrize("fmt", list(_ALPHA_PATTERN_MODULES))
def test_alpha_pattern_with_default_adapter_slot(fmt):
    # PEFT-saved Kohya files name the slot: `.lora_down.default.weight`.
    kohya, native, dit_class_name = _ALPHA_PATTERN_MODULES[fmt]
    plain = _kohya([kohya.format("v")])
    plain[f"lora_unet_{kohya.format('v')}.alpha"] = torch.tensor(float(RANK))
    slotted = {k.replace(".weight", ".default.weight"): v for k, v in plain.items()}
    cfg = {"alpha_pattern": {"to_v": RANK}}
    out = normalize_lora_state_dict(
        slotted, adapter_config=cfg, dit_class_name=dit_class_name
    )
    ref = normalize_lora_state_dict(
        plain, adapter_config=cfg, dit_class_name=dit_class_name
    )
    n = native.format("v")
    for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha"):
        torch.testing.assert_close(out[n + suffix], ref[n + suffix])


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


@pytest.mark.parametrize(
    "module",
    ["layers_0_attention_to_q", "transformer_blocks_0_img_mlp_net_0_proj"],
)
def test_negative_kohya_alpha_is_rejected_before_the_converter(module):
    # Diffusers' converters loop forever on a negative alpha / rank scale.
    sd = _kohya([module])
    sd[f"lora_unet_{module}.alpha"] = torch.tensor(-ALPHA)
    with pytest.raises(ValueError, match="must be a non-negative number"):
        normalize_lora_state_dict(sd)


def test_module_dropped_by_the_converter_fails_the_load(monkeypatch):
    # Releases that cannot split a fused qkv drop it and keep the rest, which
    # would load only the feed-forward part of the LoRA.
    convert = format_adapter.lcu._convert_non_diffusers_z_image_lora_to_diffusers

    def drops_qkv(state_dict):
        return convert({k: v for k, v in state_dict.items() if "_qkv." not in k})

    monkeypatch.setattr(
        format_adapter.lcu,
        "_convert_non_diffusers_z_image_lora_to_diffusers",
        drops_qkv,
    )
    sd = _kohya(["layers_0_attention_qkv", "layers_0_feed_forward_w1"])
    with pytest.raises(ValueError, match="drops 1 module"):
        normalize_lora_state_dict(sd)


def test_metadata_alpha_reaches_a_kohya_file_with_an_adapter_slot(tmp_path):
    # `.lora_down.default.weight` keys still count as one rank, so the global
    # alpha in the safetensors metadata applies: alpha 8 at rank 4 is 2x.
    module = "layers_0_attention_to_q"
    raw = _kohya([module])
    sd = {
        f"lora_unet_{module}.lora_down.default.weight": raw[
            f"lora_unet_{module}.lora_down.weight"
        ],
        f"lora_unet_{module}.lora_up.default.weight": raw[
            f"lora_unet_{module}.lora_up.weight"
        ],
    }
    path = tmp_path / "slot.safetensors"
    save_file(sd, str(path), metadata={"alpha": str(int(ALPHA))})
    config = load_peft_config(str(path))
    assert config["lora_alpha"] == ALPHA

    out = normalize_lora_state_dict(sd, adapter_config=config)
    got = (
        out["layers.0.attention.to_q.lora_B.weight"]
        @ out["layers.0.attention.to_q.lora_A.weight"]
    )
    ref = (
        raw[f"lora_unet_{module}.lora_up.weight"]
        @ raw[f"lora_unet_{module}.lora_down.weight"]
    )
    torch.testing.assert_close(got, (ALPHA / RANK) * ref)


def test_z_image_lora_on_adaln_only_resolves_by_loaded_dit():
    sd = _kohya(["layers_0_adaLN_modulation_0"])
    assert (
        detect_lora_format_from_state_dict(sd, "ZImageTransformer2DModel")
        == LoRAFormat.KOHYA_Z_IMAGE
    )
    out = normalize_lora_state_dict(sd, dit_class_name="ZImageTransformer2DModel")
    assert {k for k in out if k.endswith("lora_A.weight")} == {
        "layers.0.adaLN_modulation.0.lora_A.weight"
    }


def test_qwen_image_top_level_projection_loads_with_the_blocks():
    # diffusers' converter does not rename `img_in`; it is mapped here and
    # keeps its alpha, so it applies at alpha / rank like the block modules.
    sd = _kohya(["transformer_blocks_0_img_mlp_net_0_proj", "img_in", "proj_out"])
    out = normalize_lora_state_dict(sd)
    for kohya, native in (("img_in", "img_in"), ("proj_out", "proj_out")):
        got = out[f"{native}.lora_B.weight"] @ out[f"{native}.lora_A.weight"]
        ref = (
            sd[f"lora_unet_{kohya}.lora_up.weight"]
            @ sd[f"lora_unet_{kohya}.lora_down.weight"]
        )
        torch.testing.assert_close(
            got * out[f"{native}.alpha"].item() / RANK, (ALPHA / RANK) * ref
        )
    assert "transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight" in out


def test_fused_qkv_next_to_complete_split_qkv_loads_the_split_weights(monkeypatch):
    # Diffusers skips a fused qkv when the file has all of to_q/to_k/to_v.
    convert = format_adapter.lcu._convert_non_diffusers_z_image_lora_to_diffusers

    def skips_qkv(state_dict):
        return convert({k: v for k, v in state_dict.items() if "_qkv." not in k})

    monkeypatch.setattr(
        format_adapter.lcu,
        "_convert_non_diffusers_z_image_lora_to_diffusers",
        skips_qkv,
    )
    split = [f"layers_0_attention_to_{x}" for x in "qkv"]
    out = normalize_lora_state_dict(_kohya(["layers_0_attention_qkv", *split]))
    assert {k.rsplit(".", 2)[0] for k in out if k.endswith("lora_A.weight")} == {
        f"layers.0.attention.to_{x}" for x in "qkv"
    }


def test_qwen_image_top_level_only_file_resolves_by_loaded_dit():
    sd = _kohya(["img_in"])
    assert (
        detect_lora_format_from_state_dict(sd, "QwenImageTransformer2DModel")
        == LoRAFormat.KOHYA_QWEN_IMAGE
    )
    out = normalize_lora_state_dict(sd, dit_class_name="QwenImageTransformer2DModel")
    assert {k for k in out if k.endswith("lora_A.weight")} == {"img_in.lora_A.weight"}


def test_top_level_module_next_to_unreadable_block_tensors_fails():
    # LoKr block tensors next to a plain img_in pair would otherwise load only
    # img_in.
    sd = _kohya(["img_in"])
    sd["lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lokr_w1"] = torch.randn(4, 4)
    sd["lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lokr_w2"] = torch.randn(4, 4)
    with pytest.raises(ValueError, match="Kohya Qwen-Image"):
        normalize_lora_state_dict(sd, dit_class_name="QwenImageTransformer2DModel")


def test_top_level_module_alpha_must_be_one_number():
    sd = _kohya(["transformer_blocks_0_img_mlp_net_0_proj", "img_in"])
    sd["lora_unet_img_in.alpha"] = torch.tensor([ALPHA, ALPHA])
    with pytest.raises(ValueError, match="must be a single number"):
        normalize_lora_state_dict(sd)
