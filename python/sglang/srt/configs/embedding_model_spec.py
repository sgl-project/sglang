"""Embedding-model capabilities resolved from model architecture and server intent.

This module is deliberately declarative.  Server-argument resolution, model
implementations, documentation, and benchmarks can consume the same contract
without each reimplementing a partial list of embedding architectures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


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


class EmbeddingExecution(str, Enum):
    """The model path that produces an embedding."""

    NONE = "none"
    ENCODER_ONLY = "encoder_only"
    DECODER_POOLING = "decoder_pooling"
    MULTIMODAL = "multimodal"
    CLASSIFICATION = "classification"


class AttentionPattern(str, Enum):
    NONE = "none"
    BIDIRECTIONAL = "bidirectional"
    CAUSAL = "causal"


class BCGEligibility(str, Enum):
    """Whether a model has a validated BCG prefill strategy."""

    DISABLED = "disabled"
    STANDARD = "standard"
    FULL_ENCODER = "full_encoder"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Resolved capabilities for an embedding or pooling model.

    ``auto_enable_embedding`` is intentionally narrow: it is true only when a
    checkpoint unambiguously declares a pooling/embedding-only architecture.
    Decoder checkpoints trained for embeddings still require explicit user
    intent.
    """

    family: str
    task: EmbeddingTask
    execution: EmbeddingExecution
    attention: AttentionPattern
    pooling: PoolingStrategy
    normalize: bool
    postprocessor: str
    tokenizer_special_tokens: str
    supports_dimensions: bool
    supports_token_embeddings: bool
    supports_multimodal: bool
    requires_embedding_flag: bool
    auto_enable_embedding: bool
    bidirectional_attention: bool
    bcg_prefill_policy: BCGPrefillPolicy
    safe_disable_radix_cache: bool = False
    safe_disable_chunked_prefill: bool = False
    safe_disable_kv_cache: bool = False

    @property
    def bcg_eligibility(self) -> BCGEligibility:
        if self.bcg_prefill_policy == BCGPrefillPolicy.FULL_ENCODER:
            return BCGEligibility.FULL_ENCODER
        if self.task == EmbeddingTask.EMBED:
            return BCGEligibility.STANDARD
        return BCGEligibility.DISABLED

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable description of this contract."""

        return {
            "family": self.family,
            "task": self.task.value,
            "execution": self.execution.value,
            "attention": self.attention.value,
            "pooling": self.pooling.value,
            "normalize": self.normalize,
            "postprocessor": self.postprocessor,
            "tokenizer_special_tokens": self.tokenizer_special_tokens,
            "supports_dimensions": self.supports_dimensions,
            "supports_token_embeddings": self.supports_token_embeddings,
            "supports_multimodal": self.supports_multimodal,
            "requires_embedding_flag": self.requires_embedding_flag,
            "auto_enable_embedding": self.auto_enable_embedding,
            "bcg_eligibility": self.bcg_eligibility.value,
            "safe_disable_kv_cache": self.safe_disable_kv_cache,
            "safe_disable_radix_cache": self.safe_disable_radix_cache,
            "safe_disable_chunked_prefill": self.safe_disable_chunked_prefill,
        }


_EMBEDDING_ARCHITECTURES = {
    "BertModel": ("bert", EmbeddingExecution.ENCODER_ONLY, PoolingStrategy.CLS),
    "CLIPModel": ("clip", EmbeddingExecution.MULTIMODAL, PoolingStrategy.LAST),
    "Contriever": (
        "contriever",
        EmbeddingExecution.ENCODER_ONLY,
        PoolingStrategy.MODEL_DEFINED,
    ),
    "LlamaEmbeddingModel": (
        "llama_embedding",
        EmbeddingExecution.DECODER_POOLING,
        PoolingStrategy.LAST,
    ),
    "MistralModel": (
        "mistral_embedding",
        EmbeddingExecution.DECODER_POOLING,
        PoolingStrategy.LAST,
    ),
    "XLMRobertaModel": (
        "xlm_roberta",
        EmbeddingExecution.ENCODER_ONLY,
        PoolingStrategy.CLS,
    ),
}

_CLASSIFICATION_ARCHITECTURES = {
    "BertForSequenceClassification",
    "LlamaForSequenceClassification",
    "LlamaForSequenceClassificationWithNormal_Weights",
    "Qwen2ForSequenceClassification",
    "Qwen3ForSequenceClassification",
    "XLMRobertaForSequenceClassification",
}


def embedding_support_matrix() -> list[dict[str, Any]]:
    """Return registry-backed rows for docs, tooling, and CI coverage checks.

    Entries describe architecture-level guarantees.  Runtime-only properties,
    such as the available Matryoshka dimensions and captured BCG sizes, belong
    to :func:`resolved_embedding_plan`.
    """

    rows = []
    for architecture, (family, execution, pooling) in _EMBEDDING_ARCHITECTURES.items():
        spec = _native_embedding_spec(family, execution, pooling)
        rows.append({"architecture": architecture, **spec.as_dict()})
    rows.append(
        {
            "architecture": "Gemma3TextModel (use_bidirectional_attention=true)",
            **_embedding_gemma_spec().as_dict(),
        }
    )
    return rows


def _embedding_gemma_spec() -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        family="embeddinggemma",
        task=EmbeddingTask.EMBED,
        execution=EmbeddingExecution.ENCODER_ONLY,
        attention=AttentionPattern.BIDIRECTIONAL,
        pooling=PoolingStrategy.MEAN,
        normalize=True,
        postprocessor="sentence_transformers",
        tokenizer_special_tokens="model_default",
        supports_dimensions=False,
        supports_token_embeddings=False,
        supports_multimodal=False,
        requires_embedding_flag=False,
        auto_enable_embedding=True,
        bidirectional_attention=True,
        bcg_prefill_policy=BCGPrefillPolicy.FULL_ENCODER,
        safe_disable_radix_cache=True,
        safe_disable_chunked_prefill=True,
        safe_disable_kv_cache=True,
    )


def _native_embedding_spec(
    family: str, execution: EmbeddingExecution, pooling: PoolingStrategy
) -> EmbeddingModelSpec:
    return EmbeddingModelSpec(
        family=family,
        task=EmbeddingTask.EMBED,
        execution=execution,
        attention=(
            AttentionPattern.CAUSAL
            if execution == EmbeddingExecution.DECODER_POOLING
            else AttentionPattern.BIDIRECTIONAL
        ),
        pooling=pooling,
        normalize=True,
        postprocessor="model_defined",
        tokenizer_special_tokens="model_default",
        supports_dimensions=False,
        supports_token_embeddings=False,
        supports_multimodal=execution == EmbeddingExecution.MULTIMODAL,
        requires_embedding_flag=False,
        auto_enable_embedding=True,
        bidirectional_attention=execution != EmbeddingExecution.DECODER_POOLING,
        bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
    )


def resolved_embedding_plan(
    spec: EmbeddingModelSpec, *, config: Any, model_config: Any
) -> dict[str, Any]:
    """Combine static capabilities with the effective server configuration.

    This boundary deliberately accepts duck-typed arguments so the declarative
    registry remains independent of ServerArgs and ModelConfig import cycles.
    `config` must answer with the *resolved* configuration -- the readback
    callers pass `resolving_view(record)`, since the record's fields are the raw
    input.
    """

    prefill_graph = getattr(getattr(config, "cuda_graph_config", None), "prefill", None)
    backend = getattr(prefill_graph, "backend", None)
    backend_value = getattr(backend, "value", backend)
    capture_sizes = getattr(prefill_graph, "bs", None) or []
    max_capture_tokens = getattr(prefill_graph, "max_bs", None)

    return {
        **spec.as_dict(),
        "enabled": bool(getattr(config, "is_embedding", False)),
        "supports_dimensions": bool(getattr(model_config, "is_matryoshka", False)),
        "matryoshka_dimensions": list(
            getattr(model_config, "matryoshka_dimensions", None) or []
        ),
        "bcg": {
            "prefill_backend": backend_value,
            "enabled": backend_value == "breakable",
            "capture_token_budget": max_capture_tokens,
            "capture_batch_sizes": list(capture_sizes),
        },
        "cache": {
            "kv_cache_disabled": bool(
                getattr(config, "prefill_only_disable_kv_cache", False)
            ),
            "radix_cache_disabled": bool(getattr(config, "disable_radix_cache", False)),
            "chunked_prefill_disabled": getattr(config, "chunked_prefill_size", None)
            == -1,
        },
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
        return _embedding_gemma_spec()

    if architecture_set & _CLASSIFICATION_ARCHITECTURES:
        return EmbeddingModelSpec(
            family="sequence_classification",
            task=EmbeddingTask.CLASSIFY,
            execution=EmbeddingExecution.CLASSIFICATION,
            attention=AttentionPattern.NONE,
            pooling=PoolingStrategy.MODEL_DEFINED,
            normalize=False,
            postprocessor="model_defined",
            tokenizer_special_tokens="model_default",
            supports_dimensions=False,
            supports_token_embeddings=False,
            supports_multimodal=False,
            requires_embedding_flag=False,
            auto_enable_embedding=False,
            bidirectional_attention=False,
            bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
        )

    for architecture, (family, execution, pooling) in _EMBEDDING_ARCHITECTURES.items():
        if architecture in architecture_set:
            return _native_embedding_spec(family, execution, pooling)

    if is_embedding_requested:
        return EmbeddingModelSpec(
            family="explicit_decoder_embedding",
            task=EmbeddingTask.EMBED,
            execution=EmbeddingExecution.DECODER_POOLING,
            attention=AttentionPattern.CAUSAL,
            pooling=PoolingStrategy.MODEL_DEFINED,
            normalize=True,
            postprocessor="model_defined",
            tokenizer_special_tokens="model_default",
            supports_dimensions=False,
            supports_token_embeddings=False,
            supports_multimodal=False,
            requires_embedding_flag=True,
            auto_enable_embedding=False,
            bidirectional_attention=False,
            bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
        )

    return EmbeddingModelSpec(
        family="none",
        task=EmbeddingTask.NONE,
        execution=EmbeddingExecution.NONE,
        attention=AttentionPattern.NONE,
        pooling=PoolingStrategy.MODEL_DEFINED,
        normalize=False,
        postprocessor="none",
        tokenizer_special_tokens="model_default",
        supports_dimensions=False,
        supports_token_embeddings=False,
        supports_multimodal=False,
        requires_embedding_flag=False,
        auto_enable_embedding=False,
        bidirectional_attention=False,
        bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
    )
