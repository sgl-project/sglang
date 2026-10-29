"""PEFT LoRA normalization and fail-closed capability tests."""

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.flux import FluxArchConfig
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    _convert_ai_toolkit_flux_lora,
    logger,
    normalize_lora_state_dict,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.peft_adapter import (
    apply_peft_config,
    get_peft_lora_alpha,
    load_peft_config,
)


def test_peft_wrapper_slot_and_rslora_scaling_preserve_delta():
    lora_a = torch.randn(4, 8)
    lora_b = torch.randn(16, 4)
    normalized = normalize_lora_state_dict(
        {
            "base_model.model.transformer.proj.lora_A.default.weight": lora_a,
            "base_model.model.transformer.proj.lora_B.default.weight": lora_b,
        },
        adapter_config={
            "use_rslora": True,
            "lora_alpha": 8,
            "alpha_pattern": {"transformer.proj": 8},
        },
    )

    normalized_b = normalized["transformer.proj.lora_B.weight"]
    ordinary_delta = (8 / 4) * normalized_b @ lora_a
    expected_delta = (8 / math.sqrt(4)) * lora_b @ lora_a
    torch.testing.assert_close(ordinary_delta, expected_delta)
    assert normalized["transformer.proj.alpha"].item() == 8


@pytest.mark.parametrize(
    ("state_dict", "adapter_config"),
    [
        ({}, {"use_dora": True}),
        ({}, {"modules_to_save": ["head"]}),
        ({}, {"target_parameters": ["experts.weight"]}),
        ({}, {"fan_in_fan_out": True}),
        ({"encoder.lora_embedding_A.weight": torch.ones(2, 2)}, {}),
    ],
)
def test_unsupported_peft_runtime_semantics_fail_closed(state_dict, adapter_config):
    with pytest.raises(ValueError, match="not supported|unsupported"):
        normalize_lora_state_dict(state_dict, adapter_config=adapter_config)


def test_invalid_peft_lora_alpha_fails_closed():
    with pytest.raises(ValueError, match="positive integer"):
        get_peft_lora_alpha({"lora_alpha": 8.5})


def test_safetensors_alpha_metadata_supplies_peft_config(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    save_file(
        {"proj.lora_A.default.weight": torch.ones(4, 8)},
        weight_path,
        metadata={"alpha": "128"},
    )

    assert load_peft_config(str(weight_path))["lora_alpha"] == 128


def test_mixed_rank_native_safetensors_does_not_apply_global_alpha(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    save_file(
        {
            "gate.lora_A.weight": torch.ones(4, 8),
            "proj.lora_A.weight": torch.ones(8, 8),
        },
        weight_path,
        metadata={"lora_alpha": "8"},
    )

    assert "lora_alpha" not in load_peft_config(str(weight_path))


def test_diffusers_adapter_metadata_supplies_transformer_peft_config(tmp_path):
    """diffusers save_lora_weights stores alpha only in lora_adapter_metadata (e.g. r=64, alpha=128)."""
    weight_path = tmp_path / "pytorch_lora_weights.safetensors"
    packed = {
        "transformer.r": 4,
        "transformer.lora_alpha": 8,
        "transformer.alpha_pattern": {"proj": 2},
        "text_encoder.lora_alpha": 16,
    }
    save_file(
        {"transformer.proj.lora_A.weight": torch.ones(4, 8)},
        weight_path,
        metadata={"lora_adapter_metadata": json.dumps(packed)},
    )

    config = load_peft_config(str(weight_path))
    assert config["lora_alpha"] == 8
    assert config["alpha_pattern"] == {"proj": 2}
    assert "text_encoder.lora_alpha" not in config

    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4}))
    with pytest.raises(ValueError, match="conflicts with safetensors metadata"):
        load_peft_config(str(weight_path))


def test_peft_config_applies_to_keys_without_weight_suffix():
    # Kohya H3 conversion emits blocks.N.*.lora_A/B without ".weight".
    lora_b = torch.ones(8, 4)
    out = apply_peft_config(
        {"blocks.0.attn.lora_A": torch.ones(4, 8), "blocks.0.attn.lora_B": lora_b},
        {"use_rslora": True, "alpha_pattern": {"attn": 8}},
    )
    torch.testing.assert_close(out["blocks.0.attn.lora_B"], lora_b * 2.0)
    assert out["blocks.0.attn.alpha"].item() == 8


def test_ai_toolkit_flux_alpha_follows_renamed_modules():
    state = {}
    for module, rows in (
        ("double_blocks.0.img_attn.qkv", 12),
        ("double_blocks.0.img_attn.proj", 4),
        ("double_blocks.0.img_mlp.0", 8),
        ("single_blocks.0.linear1", 20),
        ("single_blocks.0.linear2", 4),
    ):
        state[f"{module}.lora_A.weight"] = torch.randn(2, 4)
        state[f"{module}.lora_B.weight"] = torch.randn(rows, 2)
        state[f"{module}.alpha"] = torch.tensor(1.0)
    out = _convert_ai_toolkit_flux_lora(state, logger)
    alphas = {k for k in out if k.endswith(".alpha")}
    assert alphas == {
        f"{target}.alpha"
        for target in (
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.attn.to_out.0",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "single_transformer_blocks.0.attn.to_out",
            "double_blocks.0.img_mlp.0",
        )
    }
    mapping = get_param_names_mapping(FluxArchConfig().param_names_mapping)
    assert {mapping(k)[0] for k in out if ".img_mlp." in k} == {
        "transformer_blocks.0.ff.linear_in.lora_A.weight",
        "transformer_blocks.0.ff.linear_in.lora_B.weight",
        "transformer_blocks.0.ff.linear_in.alpha",
    }
