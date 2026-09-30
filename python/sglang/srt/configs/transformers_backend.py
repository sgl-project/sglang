# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

from sglang.srt.configs.transformers_task import resolve_transformers_task

_MULTIMODAL_CACHE_MODELS = frozenset({"llava", "qwen2_vl", "qwen2_5_vl"})


def supports_transformers_multimodal_cache(config):
    return config.model_type in _MULTIMODAL_CACHE_MODELS and not getattr(
        config, "auto_map", None
    )


def transformers_requires_full_sequence(model_config):
    resolve_transformers_task(model_config)
    spec = model_config.embedding_model_spec
    return bool(
        spec.safe_disable_chunked_prefill
        or spec.safe_disable_radix_cache
        or (
            model_config.is_multimodal
            and not supports_transformers_multimodal_cache(model_config.hf_config)
        )
    )
