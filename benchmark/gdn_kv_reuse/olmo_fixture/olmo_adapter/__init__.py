"""Experiment-only SGLang Olmo Hybrid registration; TP=1, BF16, NoPE."""

from transformers.models.olmo_hybrid.configuration_olmo_hybrid import OlmoHybridConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.qwen3_next import Qwen3NextConfig

# Reuse the existing GDN state-layout contract, without changing model identity.
OlmoHybridConfig.linear_layer_ids = property(
    lambda c: [i for i, t in enumerate(c.layer_types) if t == "linear_attention"]
)
OlmoHybridConfig.full_attention_layer_ids = property(
    lambda c: [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
)
OlmoHybridConfig.mamba2_cache_params = Qwen3NextConfig.mamba2_cache_params
OlmoHybridConfig.sglang_olmo_hybrid_adapter = True
register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=OlmoHybridConfig,
        backend_class_name="olmo_adapter.backend.OlmoGDNAttnBackend",
        arch_names=["OlmoHybridForCausalLM"],
        support_mamba_cache_extra_buffer=True,
    )
)
