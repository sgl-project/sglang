# SPDX-License-Identifier: Apache-2.0
"""Diffusers `save_lora_weights` LoRAs (`transformer.` on every key) load onto DiT names."""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlus_2511_PipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    strip_diffusers_transformer_prefix,
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

    out = strip_diffusers_transformer_prefix(normalize_lora_state_dict(sd))

    assert set(out) == {
        f"transformer_blocks.0.{m}.lora_{ab}.weight"
        for m in ("attn.to_q", "img_mlp.net.2")
        for ab in "AB"
    }
    arch = QwenImageEditPlus_2511_PipelineConfig().dit_config.arch_config
    lora_fn = get_param_names_mapping(arch.lora_param_names_mapping)
    param_fn = get_param_names_mapping(arch.param_names_mapping)
    name, _, _ = lora_fn("transformer_blocks.0.attn.to_q.lora_A")
    assert param_fn(name)[0].startswith("transformer_blocks.0.attn.")


def test_mixed_component_prefixes_are_kept():
    # Only a file whose every key is under `transformer.` is a diffusers DiT LoRA.
    sd = _lora(
        [
            "transformer.transformer_blocks.0.attn.to_q",
            "text_encoder.encoder.layers.0.self_attn.q_proj",
        ]
    )

    assert set(strip_diffusers_transformer_prefix(sd)) == set(sd)
