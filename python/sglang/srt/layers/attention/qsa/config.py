"""Shared QSA profile parsing across model variants.

``QSAProfile`` normalizes each model family's HF-config indexer schema,
so backends, draft utilities and model glue branch on a stable variant name,
not on raw config keys. ``compressed`` is Qwen4-Exp block compression;
``tokenwise`` is qsa_0511 / Qwen3.5-DSA per-token indexing.
DeepSeek NSA configs also expose ``index_topk``,
so the tokenwise schema is additionally gated on a Qwen ``model_type``.
"""

from __future__ import annotations

from typing import Optional

import msgspec

# QSA variant names.
QSA_VARIANT_COMPRESSED = "compressed"
QSA_VARIANT_TOKENWISE = "tokenwise"

# Rotary layouts the indexer can consume.
QSA_ROPE_MROPE = "mrope"
QSA_ROPE_PLAIN = "plain"

_COMPRESSED_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
_TOKENWISE_FIELDS = (
    "index_topk",
    "index_n_heads",
    "index_kv_heads",
    "index_head_dim",
)

# fast_topk_v2 only supports these compressed block top-k widths.
_COMPRESSED_BLOCK_TOPK = frozenset({512, 2048})
# fast_topk_v2 only supports a 2048-wide tokenwise top-k.
_TOKENWISE_BUDGET = 2048


class QSAProfile(msgspec.Struct, frozen=True):
    """Normalized sparse-attention indexer description for one model."""

    variant: str  # QSA_VARIANT_COMPRESSED | QSA_VARIANT_TOKENWISE
    n_heads: int  # index query heads
    kv_heads: int  # index key/value heads
    head_dim: int  # per-head index dimension
    budget: int  # tokens selected per query row
    compress_ratio: int  # 1 for tokenwise variants
    rope_mode: str  # rotary layout the indexer expects

    @property
    def block_topk(self) -> int:
        """Compressed blocks selected per query row (== budget for tokenwise)."""

        return self.budget // self.compress_ratio


def _text_config(config):
    return getattr(config, "text_config", config)


def _is_qwen_family(config) -> bool:
    model_type = str(getattr(config, "model_type", "") or "")
    return model_type.startswith("qwen")


def _require_fields(config, fields) -> dict:
    missing = [name for name in fields if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"QSA config is missing required fields: {missing}")
    return {name: int(getattr(config, name)) for name in fields}


def _parse_compressed(text_config) -> QSAProfile:
    values = _require_fields(text_config, _COMPRESSED_FIELDS)
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"QSA config values must be positive: {values}")
    if values["indexer_kv_heads"] != 1:
        raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
    ratio = values["indexer_compress_ratio"]
    budget = values["indexer_budget"]
    if ratio < 2:
        # Padding rows carry logical length 1, which must never reach a
        # compression boundary; ratio >= 2 guarantees that.
        raise ValueError(f"QSA requires indexer_compress_ratio >= 2, got {ratio}")
    if budget % ratio != 0:
        raise ValueError(
            "indexer_budget must be divisible by indexer_compress_ratio, got "
            f"{budget} / {ratio}"
        )
    if budget // ratio not in _COMPRESSED_BLOCK_TOPK:
        raise ValueError(
            "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
            f"to be one of {sorted(_COMPRESSED_BLOCK_TOPK)}, got {budget // ratio}"
        )
    return QSAProfile(
        variant=QSA_VARIANT_COMPRESSED,
        n_heads=values["indexer_n_heads"],
        kv_heads=values["indexer_kv_heads"],
        head_dim=values["indexer_head_dim"],
        budget=budget,
        compress_ratio=ratio,
        # The compressed indexer consumes the Qwen4-Exp layer's own (m)rope.
        rope_mode=QSA_ROPE_MROPE,
    )


def _parse_tokenwise(text_config) -> QSAProfile:
    values = _require_fields(text_config, _TOKENWISE_FIELDS)
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"QSA config values must be positive: {values}")
    if values["index_topk"] != _TOKENWISE_BUDGET:
        raise ValueError(
            f"fast_topk_v2 only supports index_topk = {_TOKENWISE_BUDGET}, "
            f"got {values['index_topk']}"
        )
    if values["index_kv_heads"] != 1:
        raise ValueError(
            f"QSA tokenwise index requires index_kv_heads = 1 (MQA), "
            f"got {values['index_kv_heads']}"
        )
    return QSAProfile(
        variant=QSA_VARIANT_TOKENWISE,
        n_heads=values["index_n_heads"],
        kv_heads=values["index_kv_heads"],
        head_dim=values["index_head_dim"],
        budget=values["index_topk"],
        compress_ratio=1,
        # The tokenwise indexer owns plain per-token rotary positions.
        rope_mode=QSA_ROPE_PLAIN,
    )


def parse_qsa_profile(config) -> Optional[QSAProfile]:
    """QSA profile of config, None if absent; malformed schemas raise ValueError."""

    if config is None:
        return None
    text_config = _text_config(config)
    if text_config is None:
        return None
    has_compressed = getattr(text_config, "indexer_n_heads", None) is not None
    has_tokenwise = getattr(
        text_config, "index_topk", None
    ) is not None and _is_qwen_family(text_config)
    if has_compressed and has_tokenwise:
        raise ValueError(
            "Ambiguous QSA config: both compressed (indexer_*) and tokenwise "
            "(index_*) indexer fields are set"
        )
    if has_compressed:
        return _parse_compressed(text_config)
    if has_tokenwise:
        return _parse_tokenwise(text_config)
    return None


def is_qwen_qsa(config) -> bool:
    """Return whether the config describes a supported Qwen QSA variant."""

    return parse_qsa_profile(config) is not None


__all__ = [
    "QSAProfile",
    "QSA_ROPE_MROPE",
    "QSA_ROPE_PLAIN",
    "QSA_VARIANT_COMPRESSED",
    "QSA_VARIANT_TOKENWISE",
    "is_qwen_qsa",
    "parse_qsa_profile",
]
