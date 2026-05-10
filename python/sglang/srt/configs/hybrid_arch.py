from __future__ import annotations

from typing import Any, Optional, Union

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.model_config import ModelConfig


def qwen3_next_config(model_config: ModelConfig) -> Optional[Qwen3NextConfig]:
    config = model_config.hf_config
    if isinstance(config, Qwen3NextConfig):
        return config
    return None


def hybrid_lightning_config(model_config: ModelConfig) -> Optional[BailingHybridConfig]:
    config = model_config.hf_config
    if isinstance(config, BailingHybridConfig):
        return config
    return None


def hybrid_gdn_config(
    model_config: ModelConfig,
) -> Optional[
    Union[
        Qwen3NextConfig,
        Qwen3_5Config,
        Qwen3_5MoeConfig,
        JetNemotronConfig,
        JetVLMConfig,
    ]
]:
    config = model_config.hf_config.get_text_config()
    if isinstance(
        config,
        Qwen3NextConfig
        | Qwen3_5Config
        | Qwen3_5MoeConfig
        | JetNemotronConfig
        | JetVLMConfig,
    ):
        return config
    return None


def mamba2_config(
    model_config: ModelConfig,
    *,
    is_draft_worker: bool,
) -> Optional[
    Union[
        FalconH1Config,
        NemotronHConfig,
        Lfm2Config,
        Lfm2MoeConfig,
        Lfm2VlConfig,
        NemotronH_Nano_VL_V2_Config,
        GraniteMoeHybridConfig,
    ]
]:
    config = model_config.hf_config
    if isinstance(config, NemotronHConfig) and is_draft_worker:
        # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
        # so they shouldn't use HybridLinearAttnBackend
        pattern = getattr(config, "mtp_hybrid_override_pattern", None)
        if pattern is not None and "M" not in pattern:
            return None
    if isinstance(
        config,
        FalconH1Config | NemotronHConfig | Lfm2Config | Lfm2MoeConfig | Lfm2VlConfig,
    ):
        return config
    if isinstance(config, NemotronH_Nano_VL_V2_Config):
        return config.llm_config

    if isinstance(config, GraniteMoeHybridConfig):
        has_mamba = any(
            layer_type == "mamba" for layer_type in getattr(config, "layer_types", [])
        )
        if not has_mamba:
            return None
        else:
            return config

    return None


def kimi_linear_config(model_config: ModelConfig) -> Optional[KimiLinearConfig]:
    config = model_config.hf_config
    if isinstance(config, KimiLinearConfig):
        return config
    return None


def linear_attn_model_spec(model_config: ModelConfig) -> Optional[Any]:
    result = get_linear_attn_config(model_config.hf_config)
    return result[0] if result else None


def mambaish_config(
    model_config: ModelConfig,
    *,
    is_draft_worker: bool,
) -> Optional[Any]:
    existing = (
        mamba2_config(model_config, is_draft_worker=is_draft_worker)
        or hybrid_gdn_config(model_config)
        or kimi_linear_config(model_config)
        or hybrid_lightning_config(model_config)
    )
    if existing:
        return existing
    result = get_linear_attn_config(model_config.hf_config)
    return result[1] if result else None
