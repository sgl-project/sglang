"""Embedding-model capabilities resolved from model architecture and server intent.

This module is deliberately declarative.  Server-argument resolution, model
implementations, documentation, and benchmarks can consume the same contract
without each reimplementing a partial list of embedding architectures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EmbeddingTask(str, Enum):
    NONE = "none"
    EMBED = "embed"
    CLASSIFY = "classify"


class PoolingStrategy(str, Enum):
    MODEL_DEFINED = "model_defined"
    CLS = "cls"
    LAST = "last"
    MEAN = "mean"


class BCGPrefillPolicy(str, Enum):
    """Whether a model has a validated breakable-CUDA-graph prefill policy."""

    DEFAULT = "default"
    FULL_ENCODER = "full_encoder"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Resolved capabilities for an embedding or pooling model.

    ``auto_enable_embedding`` is intentionally narrow: it is true only when a
    checkpoint unambiguously declares an embedding-only architecture. Decoder
    checkpoints trained for embeddings still require explicit user intent.
    """

    family: str
    task: EmbeddingTask
    pooling: PoolingStrategy
    normalize: bool
    requires_embedding_flag: bool
    auto_enable_embedding: bool
    bidirectional_attention: bool
    bcg_prefill_policy: BCGPrefillPolicy
    safe_disable_radix_cache: bool = False
    safe_disable_chunked_prefill: bool = False
    safe_disable_kv_cache: bool = False


_EMBEDDING_ARCHITECTURES = {
    "BertModel": ("bert", PoolingStrategy.CLS),
    "CLIPModel": ("clip", PoolingStrategy.LAST),
    "Contriever": ("contriever", PoolingStrategy.MODEL_DEFINED),
    "LlamaEmbeddingModel": ("llama_embedding", PoolingStrategy.LAST),
    "MistralModel": ("mistral_embedding", PoolingStrategy.LAST),
    "XLMRobertaModel": ("xlm_roberta", PoolingStrategy.CLS),
}

_CLASSIFICATION_ARCHITECTURES = {
    "BertForSequenceClassification",
    "LlamaForSequenceClassification",
    "LlamaForSequenceClassificationWithNormal_Weights",
    "Qwen2ForSequenceClassification",
    "Qwen3ForSequenceClassification",
    "XLMRobertaForSequenceClassification",
}


def resolve_embedding_model_spec(
    architectures: Sequence[str] | None,
    *,
    is_embedding_requested: bool,
    is_embedding_gemma: bool,
) -> EmbeddingModelSpec:
    """Resolve a conservative embedding capability description.

    Unknown architectures remain ``NONE`` unless the caller explicitly asks
    for embedding mode.  This preserves today's CausalLM behavior while
    allowing model-specific support to be added in one location.
    """

    architecture_set = set(architectures or ())

    if is_embedding_gemma:
        return EmbeddingModelSpec(
            family="embeddinggemma",
            task=EmbeddingTask.EMBED,
            pooling=PoolingStrategy.MEAN,
            normalize=True,
            requires_embedding_flag=False,
            auto_enable_embedding=True,
            bidirectional_attention=True,
            bcg_prefill_policy=BCGPrefillPolicy.FULL_ENCODER,
            safe_disable_radix_cache=True,
            safe_disable_chunked_prefill=True,
            safe_disable_kv_cache=True,
        )

    if architecture_set & _CLASSIFICATION_ARCHITECTURES:
        return EmbeddingModelSpec(
            family="sequence_classification",
            task=EmbeddingTask.CLASSIFY,
            pooling=PoolingStrategy.MODEL_DEFINED,
            normalize=False,
            requires_embedding_flag=False,
            auto_enable_embedding=False,
            bidirectional_attention=False,
            bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
        )

    for architecture, (family, pooling) in _EMBEDDING_ARCHITECTURES.items():
        if architecture in architecture_set:
            return EmbeddingModelSpec(
                family=family,
                task=EmbeddingTask.EMBED,
                pooling=pooling,
                normalize=True,
                requires_embedding_flag=True,
                auto_enable_embedding=False,
                bidirectional_attention=False,
                bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
            )

    if is_embedding_requested:
        return EmbeddingModelSpec(
            family="explicit_decoder_embedding",
            task=EmbeddingTask.EMBED,
            pooling=PoolingStrategy.MODEL_DEFINED,
            normalize=True,
            requires_embedding_flag=True,
            auto_enable_embedding=False,
            bidirectional_attention=False,
            bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
        )

    return EmbeddingModelSpec(
        family="none",
        task=EmbeddingTask.NONE,
        pooling=PoolingStrategy.MODEL_DEFINED,
        normalize=False,
        requires_embedding_flag=False,
        auto_enable_embedding=False,
        bidirectional_attention=False,
        bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
    )
