# SPDX-License-Identifier: Apache-2.0
"""Kohya/musubi `lora_unet_*` LoRAs for Qwen-Image and Z-Image normalize to native names."""

import pytest
import torch

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

    assert set(out) == {f"{n}.lora_{ab}.weight" for n in expected for ab in "AB"}
    # alpha / rank is folded into A/B, so the adapter needs no alpha key.
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
