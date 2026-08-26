from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_1_Fun_1_3B_InP_Config,
    Wan2_1_T2V_1_3B_Config,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.registry import _get_config_info


def test_wan_prompt_embed_accessors_return_transformer_tensor():
    prompt_embeds = torch.empty(1, 2, 3)
    negative_prompt_embeds = torch.empty(1, 2, 3)
    batch = SimpleNamespace(
        prompt_embeds=[prompt_embeds],
        negative_prompt_embeds=[negative_prompt_embeds],
    )

    config = WanT2V480PConfig()

    assert config.get_pos_prompt_embeds(batch) is prompt_embeds
    assert config.get_neg_prompt_embeds(batch) is negative_prompt_embeds


def test_wan21_1_3b_exact_path_uses_checkpoint_specific_config():
    _get_config_info.cache_clear()

    info = _get_config_info("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    assert info is not None
    assert info.pipeline_config_cls is Wan2_1_T2V_1_3B_Config


def test_wan21_fun_1_3b_exact_path_uses_checkpoint_specific_config():
    _get_config_info.cache_clear()

    info = _get_config_info("weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers")

    assert info is not None
    assert info.pipeline_config_cls is Wan2_1_Fun_1_3B_InP_Config


def test_unregistered_wan_pipeline_keeps_generic_config():
    _get_config_info.cache_clear()
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={"_class_name": "WanPipeline"},
    ):
        info = _get_config_info("example-org/unregistered-wan-checkpoint")
    _get_config_info.cache_clear()

    assert info is not None
    assert info.pipeline_config_cls is WanT2V480PConfig
