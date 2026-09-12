"""QSA profile parsing for Qwen4-Exp compressed indexing."""

from __future__ import annotations

from typing import Optional

import msgspec

_COMPRESSED_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
# fast_topk_v2 only supports these compressed block top-k widths.
_COMPRESSED_BLOCK_TOPK = frozenset({512, 2048})


class QSAProfile(msgspec.Struct, frozen=True):
    """Compressed sparse-attention indexer configuration."""

    n_heads: int  # index query heads
    kv_heads: int  # index key/value heads
    head_dim: int  # per-head index dimension
    budget: int  # tokens selected per query row
    compress_ratio: int

    @property
    def block_topk(self) -> int:
        """Compressed blocks selected per query row."""

        return self.budget // self.compress_ratio


def _text_config(config):
    return getattr(config, "text_config", config)


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
        n_heads=values["indexer_n_heads"],
        kv_heads=values["indexer_kv_heads"],
        head_dim=values["indexer_head_dim"],
        budget=budget,
        compress_ratio=ratio,
    )


def parse_qsa_profile(config) -> Optional[QSAProfile]:
    """QSA profile of config, None if absent; malformed schemas raise ValueError."""

    if config is None:
        return None
    text_config = _text_config(config)
    if text_config is None:
        return None
    if getattr(text_config, "indexer_n_heads", None) is not None:
        return _parse_compressed(text_config)
    return None


def is_qwen_qsa(config) -> bool:
    """Return whether the config describes Qwen compressed QSA."""

    return parse_qsa_profile(config) is not None


__all__ = [
    "QSAProfile",
    "is_qwen_qsa",
    "parse_qsa_profile",
]
