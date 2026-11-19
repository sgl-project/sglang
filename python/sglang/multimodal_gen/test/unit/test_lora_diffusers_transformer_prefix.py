# SPDX-License-Identifier: Apache-2.0
"""Diffusers `save_lora_weights` LoRAs (`transformer.` on every key) normalize to DiT names."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    normalize_lora_state_dict,
)


def _lora(names: list[str]) -> dict[str, torch.Tensor]:
    return {f"{n}.lora_{ab}.weight": torch.randn(4, 4) for n in names for ab in "AB"}


def test_uniform_transformer_prefix_is_dropped():
    sd = _lora(
        [
            "transformer.transformer_blocks.0.attn.to_q",
            "transformer.transformer_blocks.0.img_mlp.net.2",
        ]
    )

    out = normalize_lora_state_dict(sd)

    assert set(out) == {
        f"transformer_blocks.0.{m}.lora_{ab}.weight"
        for m in ("attn.to_q", "img_mlp.net.2")
        for ab in "AB"
    }


def test_mixed_component_prefixes_are_kept():
    # Only a file whose every key is under `transformer.` is a diffusers DiT LoRA.
    sd = _lora(
        [
            "transformer.transformer_blocks.0.attn.to_q",
            "text_encoder.encoder.layers.0.self_attn.q_proj",
        ]
    )

    assert set(normalize_lora_state_dict(sd)) == set(sd)
