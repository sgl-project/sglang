from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InklingMMConfig,
    InklingModelConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    MiniCPMHybridConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    ZayaConfig,
)

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
        | ZayaConfig,
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

    # Pure Mamba2 (Mamba2ForCausalLM); the flag is set in ModelConfig.
    if getattr(config, "_is_pure_mamba2", False):
        # Mamba2AttnBackend expects a mamba_chunk_size alias.
        if not hasattr(config, "mamba_chunk_size"):
            config.mamba_chunk_size = config.chunk_size

        # Build cache params here, where the runtime tp_size is available.
        if not hasattr(config, "mamba2_cache_params"):
            from sglang.srt.configs.mamba_utils import (
                Mamba2CacheParams,
                Mamba2StateShape,
            )
            from sglang.srt.runtime_context import get_parallel

            tp_size = get_parallel().tp_size if get_parallel() else 1

            state_shape = Mamba2StateShape.create(
                tp_world_size=tp_size,
                intermediate_size=config.intermediate_size,
                n_groups=config.n_groups,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                state_size=config.state_size,
                conv_kernel=config.conv_kernel,
            )
            config.mamba2_cache_params = Mamba2CacheParams(
                shape=state_shape,
                layers=list(range(config.num_hidden_layers)),
            )
        return config

    # Pure Mamba-1 (Falcon-Mamba, state-spaces Mamba); the flag is set in
    # ModelConfig. Mamba-1 uses the Mamba2 backend via a full-rank (head_dim==1)
    # state layout.
    if getattr(config, "_is_pure_mamba1", False):
        # Mamba2AttnBackend reads mamba_chunk_size; keep the conv window below it.
        if not hasattr(config, "mamba_chunk_size"):
            config.mamba_chunk_size = 256

        if not hasattr(config, "mamba2_cache_params"):
            from sglang.srt.configs.mamba_utils import (
                Mamba2CacheParams,
                Mamba2StateShape,
            )
            from sglang.srt.runtime_context import get_parallel

            tp_size = get_parallel().tp_size if get_parallel() else 1

            state_shape = Mamba2StateShape.create_mamba1(
                tp_world_size=tp_size,
                intermediate_size=config.intermediate_size,
                state_size=config.state_size,
                conv_kernel=config.conv_kernel,
            )
            config.mamba2_cache_params = Mamba2CacheParams(
                shape=state_shape,
                layers=list(range(config.num_hidden_layers)),
            )
        return config

    return None


def kimi_linear_config(model_config: ModelConfig):
    config = model_config.hf_config
    if isinstance(config, KimiLinearConfig):
        return config
    if isinstance(config, BailingHybridConfig) and config.use_kda:
        return config
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


def linear_attn_model_spec(model_config: ModelConfig):
    result = _get_linear_attn_registry_result(model_config)
    return result[0] if result else None


def mambaish_config(model_config: ModelConfig):
    existing = (
        mamba2_config(model_config)
        or hybrid_gdn_config(model_config)
        or kimi_linear_config(model_config)
        or glm5_next_config(model_config)
        or hybrid_lightning_config(model_config)
    )
    if existing:
        return existing
    result = _get_linear_attn_registry_result(model_config)
    return result[1] if result else None
