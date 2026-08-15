# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from sglang.srt.configs.neo_chat import NEOChatConfig
from sglang.srt.models.neo_chat import _stacked_weight_target
from transformers import AutoConfig

MODEL_PATH = Path("/mnt/afs/fanyijiat/models/SenseNova-U1-8B-MoT-Interleaved-bd39")


def test_neo_chat_config_loads_without_remote_code() -> None:
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)

    assert isinstance(config, NEOChatConfig)
    assert config.architectures == ["NEOChatModel"]
    assert config.model_type == "neo_chat"
    assert config.model_is_mrope is True
    assert config.llm_config.hidden_size == 4096
    assert config.llm_config.num_hidden_layers == 42
    assert config.llm_config.rope_theta == 5000000
    assert config.llm_config.rope_theta_hw == 10000
    assert config.vision_config.patch_size == 16
    assert config.vision_config.downsample_ratio == 0.5


def test_neo_chat_qkv_weight_targets_cover_both_towers() -> None:
    cases = {
        "language_model.model.layers.0.self_attn.q_proj.weight": (
            "language_model.model.layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        "language_model.model.layers.0.self_attn.k_proj_mot_gen.weight": (
            "language_model.model.layers.0.self_attn.qkv_proj_mot_gen.weight",
            "k",
        ),
        "language_model.model.layers.0.mlp.gate_proj.weight": (
            "language_model.model.layers.0.mlp.gate_proj.weight",
            None,
        ),
    }

    for source, expected in cases.items():
        assert _stacked_weight_target(source) == expected


def test_neo_chat_checkpoint_language_weight_inventory() -> None:
    index = json.loads(
        (MODEL_PATH / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    language_keys = sorted(
        key for key in index["weight_map"] if key.startswith("language_model.")
    )
    native_targets = {_stacked_weight_target(key)[0] for key in language_keys}

    assert len(language_keys) == 1096
    assert len(native_targets) == 928
