# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Platform checks for the supported Nemotron 3.5 reranker layout."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import register_model_override
from sglang.srt.configs.nemotron35_reranker import is_nemotron35_reranker_config
from sglang.srt.runtime_context import get_platform


@register_model_override("Mistral3ForSequenceClassification")
def _nemotron35_reranker_overrides(server_args: Any, hf_config: Any) -> dict:
    if get_platform().is_hip and is_nemotron35_reranker_config(hf_config.to_dict()):
        # AITER prefill is causal; other ROCm backends are not validated here.
        raise ValueError(
            "Nemotron 3.5 Rerank is not supported on ROCm; "
            "bidirectional attention has not been validated on this platform."
        )
    return {}
