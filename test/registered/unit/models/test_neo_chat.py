# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from array import array
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from sglang.srt.configs.neo_chat import NEOChatConfig
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.neo_chat import (
    NEOChatModel,
    _flow_weight_target,
    _stacked_weight_target,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import AutoConfig

_MODEL_PATH_VALUE = os.environ.get("SENSENOVA_U1_MODEL_PATH")
MODEL_PATH = Path(_MODEL_PATH_VALUE) if _MODEL_PATH_VALUE else None


def test_neo_chat_import_does_not_load_official_modeling() -> None:
    blocked = sorted(
        name
        for name in sys.modules
        if name.startswith("sensenova_u1.") and ".modeling" in name
    )

    assert blocked == []


def test_neo_chat_config_loads_without_remote_code(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["NEOChatModel"],
                "auto_map": {
                    "AutoConfig": "configuration_neo_chat.NEOChatConfig",
                },
                "model_type": "neo_chat",
                "llm_config": {
                    "attention_bias": False,
                    "head_dim": 128,
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "max_position_embeddings": 262144,
                    "max_position_embeddings_hw": 10000,
                    "num_attention_heads": 32,
                    "num_hidden_layers": 42,
                    "num_key_value_heads": 8,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 5000000,
                    "rope_theta_hw": 10000,
                    "vocab_size": 151936,
                },
                "vision_config": {
                    "auto_map": {
                        "AutoConfig": "configuration_neo_vit.NEOVisionConfig",
                    },
                    "downsample_ratio": 0.5,
                    "hidden_size": 1024,
                    "llm_hidden_size": 4096,
                    "max_position_embeddings_vision": 10000,
                    "num_channels": 3,
                    "patch_size": 16,
                    "rope_theta_vision": 10000,
                },
                "patch_size": 16,
                "downsample_ratio": 0.5,
                "template": "neo1_0",
            }
        ),
        encoding="utf-8",
    )

    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

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


def test_neo_chat_flow_weight_targets_cover_native_vision() -> None:
    assert (
        _flow_weight_target(
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight"
        )
        == "fm_modules.vision_model_mot_gen.patch_embedding.weight"
    )
    assert (
        _flow_weight_target("fm_modules.timestep_embedder.mlp.0.weight")
        == "fm_modules.timestep_embedder.mlp.0.weight"
    )


def test_neo_chat_flow_request_captures_batch_isolation_key() -> None:
    sampling_params = SamplingParams(
        max_new_tokens=1,
        custom_params={
            "__sglang_batch_isolation_key": "u1-flow:test",
            "__sglang_radix_cache_prefix_limit": 7,
        },
    )
    req = Req(
        rid="u1-flow-test",
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=sampling_params,
        vocab_size=32,
    )

    assert req.batch_isolation_key == "u1-flow:test"
    assert req._compute_max_prefix_len(20) == 7


def test_neo_chat_reuses_bounded_flow_timestep_embeddings() -> None:
    class CountingEmbedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return values[:, None].repeat(1, 4)

    class FakeFlowModules(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.timestep_embedder = CountingEmbedder()
            self.add_noise_scale_embedding = False

    model = object.__new__(NEOChatModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(noise_scale_max_value=1.0)
    model.fm_modules = FakeFlowModules()
    model._flow_timestep_embed_cache = OrderedDict()
    timesteps = torch.tensor([0.0, 0.5, 1.0])

    first, first_hit = model._flow_timestep_embeds(
        timesteps=timesteps,
        image_token_count=3,
        noise_scale=1.0,
    )
    second, second_hit = model._flow_timestep_embeds(
        timesteps=timesteps,
        image_token_count=3,
        noise_scale=1.0,
    )

    assert not first_hit
    assert second_hit
    assert model.fm_modules.timestep_embedder.calls == 2
    assert first[0] is second[0]
    assert len(model._flow_timestep_embed_cache) == 1


def test_neo_chat_batches_exact_interleave_through_regular_decode(
    monkeypatch,
) -> None:
    class FakeLanguageModel(nn.Module):
        def forward(self, *_args, **_kwargs):
            return "batched-text"

    model = object.__new__(NEOChatModel)
    nn.Module.__init__(model)
    model.language_model = FakeLanguageModel()
    forward_batch = SimpleNamespace(
        batch_size=2,
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: True,
        ),
        model_specific_states={
            "image_gen_indicators": torch.zeros(2, dtype=torch.bool),
        },
        contains_mm_inputs=lambda: False,
    )
    monkeypatch.setattr(
        NEOChatModel,
        "_exact_text_specs",
        staticmethod(lambda _forward_batch: [{}, {}]),
    )
    monkeypatch.setattr(
        NEOChatModel,
        "_install_hybrid_mask",
        lambda _self, _forward_batch: None,
    )
    monkeypatch.setattr(
        NEOChatModel,
        "_forward_exact_text",
        lambda *_args, **_kwargs: pytest.fail(
            "batched exact interleave must use regular continuous decode"
        ),
    )

    output = model.forward(
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        forward_batch,
    )

    assert output == "batched-text"


@pytest.mark.skipif(
    MODEL_PATH is None or not (MODEL_PATH / "model.safetensors.index.json").exists(),
    reason="SENSENOVA_U1_MODEL_PATH must point to a local checkpoint",
)
def test_neo_chat_checkpoint_language_weight_inventory() -> None:
    assert MODEL_PATH is not None
    index = json.loads(
        (MODEL_PATH / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    language_keys = sorted(
        key for key in index["weight_map"] if key.startswith("language_model.")
    )
    native_targets = {_stacked_weight_target(key)[0] for key in language_keys}
    flow_keys = sorted(
        key for key in index["weight_map"] if key.startswith("fm_modules.")
    )
    flow_targets = {_flow_weight_target(key) for key in flow_keys}

    assert len(language_keys) == 1096
    assert len(native_targets) == 928
    assert len(flow_keys) == 16
    assert len(flow_targets) == 16
