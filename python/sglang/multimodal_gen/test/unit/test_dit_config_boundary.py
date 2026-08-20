from dataclasses import fields

import torch

from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerArchConfig,
)
from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.models.bridges.mova_dual_tower import (
    DualTowerConditionalBridge,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.test.server.test_server_utils import (
    is_missing_diffusers_pipeline_error,
)


class _TestDiT(CachableDiT):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states


def test_dit_runtime_keeps_architecture_and_component_config_separate():
    component_config = DiTConfig(prefix="Wan")
    model = _TestDiT(config=component_config, hf_config={})

    assert model.config is component_config.arch_config
    assert model.prefix == "Wan"
    assert model._supports_cfg_cache
    assert model._spectrum_supports_cfg_cache


def test_dit_arch_config_excludes_runtime_capabilities():
    field_names = {field.name for field in fields(DiTArchConfig)}

    assert "_fsdp_shard_conditions" not in field_names
    assert "_compile_conditions" not in field_names
    assert "_supported_attention_backends" not in field_names


def test_mova_bridge_declares_runtime_capabilities_on_model_class():
    assert "_fsdp_shard_conditions" not in {
        field.name for field in fields(MOVADualTowerArchConfig)
    }
    assert DualTowerConditionalBridge._fsdp_shard_conditions
    assert DualTowerConditionalBridge._compile_conditions
    assert DualTowerConditionalBridge._supported_attention_backends


def test_server_startup_skip_only_matches_missing_diffusers_pipeline():
    assert is_missing_diffusers_pipeline_error(
        "AttributeError: module 'diffusers' has no attribute 'MOVA'"
    )
    assert is_missing_diffusers_pipeline_error(
        "Pipeline class MOVA not found in diffusers"
    )
    assert not is_missing_diffusers_pipeline_error(
        "AttributeError: MOVADualTowerConfig has no attribute '_compile_conditions'\n"
        "Loading pipeline modules from diffusers config"
    )
