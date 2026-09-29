# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

from importlib import import_module

from .utils import can_enable_torch_compile, maybe_prefix

_WRAPPERS = frozenset(
    {
        "TransformersForCausalLM",
        "TransformersMoEForCausalLM",
        "TransformersMultiModalForCausalLM",
        "TransformersMultiModalMoEForCausalLM",
        "TransformersEmbeddingModel",
        "TransformersMoEEmbeddingModel",
        "TransformersMultiModalEmbeddingModel",
        "TransformersMultiModalMoEEmbeddingModel",
        "TransformersForSequenceClassification",
        "TransformersMoEForSequenceClassification",
        "TransformersMultiModalForSequenceClassification",
        "TransformersMultiModalMoEForSequenceClassification",
        "EntryClass",
    }
)
_EXPORT_MODULES = {
    "TransformersBase": "base",
    "TransformersFusedMoE": "moe",
    "MoEMixin": "moe",
    "MultiModalMixin": "multimodal",
    "EmbeddingMixin": "pooling",
    "ClassificationMixin": "pooling",
    "CausalMixin": "causal",
    "sglang_flash_attention_forward": "attention",
}


def __getattr__(name):
    module = "modeling" if name in _WRAPPERS else _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
