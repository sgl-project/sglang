from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.configs.falcon_h1 import FalconH1Config
from sglang.srt.configs.granitemoehybrid import GraniteMoeHybridConfig
from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig
from sglang.srt.configs.interns2preview import InternS2PreviewConfig
from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.configs.jet_vlm import JetVLMConfig
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.configs.lfm2_moe import Lfm2MoeConfig
from sglang.srt.configs.lfm2_vl import Lfm2VlConfig
from sglang.srt.configs.mamba import FalconMambaConfig, MambaConfig
from sglang.srt.configs.mamba2 import Mamba2Config
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.zaya import ZayaConfig
from sglang.srt.utils.hf_transformers.common import get_hf_text_config

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def _get_linear_attn_registry_result(model_config: ModelConfig) -> Any:
    return model_config.linear_attn_registry_result


def qwen3_next_config(model_config: ModelConfig):
    config = model_config.hf_config
    if isinstance(config, Qwen3NextConfig):
        return config
    return None


def hybrid_lightning_config(model_config: ModelConfig):
    config = model_config.hf_config
    if isinstance(config, BailingHybridConfig) and not config.use_kda:
        return config
    if isinstance(config, MiniCPMHybridConfig) and config.has_lightning_layers:
        return config
    return None


def hybrid_gdn_config(model_config: ModelConfig):
    config = model_config.hf_config.get_text_config()
    if isinstance(
        config,
        Qwen3NextConfig
        | Qwen3_5Config
        | Qwen3_5MoeConfig
        | InternS2PreviewConfig
        | JetNemotronConfig
        | JetVLMConfig,
    ):
        return config
    return None


def mamba2_config(model_config: ModelConfig):
    config = model_config.hf_config
    if isinstance(config, NemotronHConfig) and model_config.is_draft_model:
        # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
        # so they shouldn't use HybridLinearAttnBackend
        pattern = getattr(config, "mtp_hybrid_override_pattern", None)
        if pattern is not None and "M" not in pattern:
            return None
    if isinstance(
        config,
        FalconH1Config
        | NemotronHConfig
        | Lfm2Config
        | Lfm2MoeConfig
        | Lfm2VlConfig
        | ZayaConfig
        | Mamba2Config
        | MambaConfig
        | FalconMambaConfig,
    ):
        return config
    if isinstance(config, InklingModelConfig):
        return config if config.mamba2_cache_params is not None else None
    if isinstance(config, InklingMMConfig):
        text_config = config.text_config
        return text_config if text_config.mamba2_cache_params is not None else None
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


def kimi_linear_config(model_config: ModelConfig):
    config = model_config.hf_config
    if isinstance(config, KimiLinearConfig):
        return config
    if isinstance(config, BailingHybridConfig) and config.use_kda:
        return config
    text_config = get_hf_text_config(config)
    if isinstance(text_config, BailingHybridConfig) and text_config.use_kda:
        return text_config
    text_config = getattr(config, "text_config", None)
    if isinstance(text_config, KimiLinearConfig):
        return text_config
    return None


def glm5_next_config(model_config: ModelConfig):
    hf_config = model_config.hf_config
    if (
        getattr(hf_config, "model_type", None) == "glm5_next"
        and not model_config.is_draft_model
    ):
        return hf_config.get_text_config()
    return None


def hybrid_kda_config(model_config: ModelConfig):
    return kimi_linear_config(model_config) or glm5_next_config(model_config)


def linear_attn_model_spec(model_config: ModelConfig):
    result = _get_linear_attn_registry_result(model_config)
    return result[0] if result else None


def mambaish_config(model_config: ModelConfig):
    existing = (
        mamba2_config(model_config)
        or hybrid_gdn_config(model_config)
        or hybrid_kda_config(model_config)
        or hybrid_lightning_config(model_config)
    )
    if existing:
        return existing
    result = _get_linear_attn_registry_result(model_config)
    return result[1] if result else None
