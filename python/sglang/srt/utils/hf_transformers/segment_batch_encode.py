# Copyright 2026 Contributors to the SGLang segment-batch-encode proposal.
# Licensed under the Apache License, Version 2.0.
"""Lossless segmented parallel encoding for long single-string prompts.

Split text on a delimiter (default: ``### Passage `` for RAG, else eos),
encode segments via Rust ``encode_batch``, concat token ids. Lossless when the
delimiter is a hard BPE boundary.
"""

from __future__ import annotations

import array
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

DEFAULT_PASSAGE_DELIMITER = "### Passage "
_DEFAULT_CACHE_MAX_ENTRIES = 100_000
_DEFAULT_CACHE_MAX_TOKENS = 8_000_000
_DEFAULT_MIN_CHARS = 4096


def _segment_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


class _SegmentIdsCache:
    def __init__(self, max_entries: int, max_total_tokens: int) -> None:
        self._max_entries = max_entries
        self._max_total_tokens = max_total_tokens
        self._data: OrderedDict[bytes, array.array] = OrderedDict()
        self._total_tokens = 0
        self._lock = threading.Lock()

    def get(self, key: bytes) -> array.array | None:
        with self._lock:
            ids = self._data.get(key)
            if ids is not None:
                self._data.move_to_end(key)
            return ids

    def put(self, key: bytes, ids: array.array) -> None:
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._total_tokens -= len(old)
            if len(ids) > self._max_total_tokens:
                return
            self._data[key] = ids
            self._total_tokens += len(ids)
            while (
                len(self._data) > self._max_entries
                or self._total_tokens > self._max_total_tokens
            ):
                _, evicted = self._data.popitem(last=False)
                self._total_tokens -= len(evicted)


def resolve_split_delimiter(
    tokenizer: PreTrainedTokenizer,
    configured: Optional[str],
    sample_text: Optional[str] = None,
) -> Optional[str]:
    if configured:
        return configured
    if sample_text and DEFAULT_PASSAGE_DELIMITER in sample_text:
        return DEFAULT_PASSAGE_DELIMITER
    return tokenizer.eos_token or None


def _concat_items(items: list[str], delimiter: str) -> str:
    if not items:
        return ""
    out = items[0]
    for item in items[1:]:
        out += delimiter + item
    return out


def _estimate_tokens(text: str) -> int:
    # chars/4 is enough for segment planning; avoids a full encode.
    return max(1, len(text) // 4)


def compute_max_workers(est_tokens: int) -> int:
    """Fan-out cap scales with length: 2 workers per 1k estimated tokens."""
    return max(1, (est_tokens * 2) // 1000)


def _plan_segment_items(
    items: list[str],
    delimiter: str,
    *,
    est_tokens: int,
    max_workers: int | None = None,
) -> list[str]:
    """Merge passages into at most ``max_workers`` segments (default: len/1k*2)."""
    if len(items) <= 1:
        return items
    n = max_workers if max_workers is not None else compute_max_workers(est_tokens)
    n = min(len(items), max(1, n))
    if n >= len(items):
        return items
    group = (len(items) + n - 1) // n
    return [_concat_items(items[i : i + group], delimiter) for i in range(0, len(items), group)]


def _encode_segments(
    tokenizer: PreTrainedTokenizerFast,
    items: list[str],
    cache: _SegmentIdsCache | None,
    add_special_tokens: bool,
) -> list[list[int]]:
    backend = tokenizer.backend_tokenizer
    if cache is None:
        encs = backend.encode_batch(items, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encs]

    digests = [_segment_digest(s) for s in items]
    seg_ids: list[list[int] | None] = [None] * len(items)
    cached: list[array.array | None] = [cache.get(d) for d in digests]
    miss = [i for i, c in enumerate(cached) if c is None]
    for i, c in enumerate(cached):
        if c is not None:
            seg_ids[i] = list(c)
    if miss:
        encs = backend.encode_batch(
            [items[i] for i in miss], add_special_tokens=add_special_tokens
        )
        for j, enc in zip(miss, encs):
            ids = array.array("i", enc.ids)
            seg_ids[j] = list(ids)
            cache.put(digests[j], ids)
    return [ids or [] for ids in seg_ids]


def segment_batch_encode_ids(
    tokenizer: PreTrainedTokenizer,
    text: str,
    *,
    split_delimiter: str,
    add_special_tokens: bool = False,
    segment_cache: _SegmentIdsCache | None = None,
    interleave_delimiter: bool = True,
) -> list[int]:
    """Encode ``text`` via segmented ``encode_batch`` when beneficial."""
    if not isinstance(text, str) or not text:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    if add_special_tokens and tokenizer.num_special_tokens_to_add() > 0:
        return tokenizer.encode(text, add_special_tokens=True)

    raw_items = text.split(split_delimiter)
    ends_with_delim = text.endswith(split_delimiter)
    if raw_items and raw_items[-1] == "":
        raw_items.pop()
    if len(raw_items) <= 1:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    est = _estimate_tokens(text)
    items = _plan_segment_items(raw_items, split_delimiter, est_tokens=est)
    if len(items) <= 1:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    delim_ids = list(tokenizer.encode(split_delimiter, add_special_tokens=False))
    seg_ids = _encode_segments(tokenizer, items, segment_cache, add_special_tokens)

    result: list[int] = []
    for i, ids in enumerate(seg_ids):
        if i > 0:
            result.extend(delim_ids)
        result.extend(ids)
    if ends_with_delim:
        result.extend(delim_ids)
    return result


def make_segment_batch_encode_tokenizer(
    tokenizer: PreTrainedTokenizer,
    *,
    split_delimiter: Optional[str] = None,
    segment_cache: bool = False,
    cache_max_entries: int = _DEFAULT_CACHE_MAX_ENTRIES,
    cache_max_tokens: int = _DEFAULT_CACHE_MAX_TOKENS,
    min_chars: int = _DEFAULT_MIN_CHARS,
) -> PreTrainedTokenizer:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "enable_segment_batch_encode requires a fast tokenizer; got %s",
            type(tokenizer).__name__,
        )
        return tokenizer

    delim = resolve_split_delimiter(tokenizer, split_delimiter)
    if not delim:
        logger.warning(
            "enable_segment_batch_encode: no delimiter; feature disabled for this tokenizer"
        )
        return tokenizer

    cache = (
        _SegmentIdsCache(cache_max_entries, cache_max_tokens) if segment_cache else None
    )
    base_cls = tokenizer.__class__

    class SegmentBatchEncodeTokenizerImpl(base_cls):  # type: ignore[misc,valid-type]
        def encode(  # type: ignore[override]
            self,
            text,
            *args,
            add_special_tokens: bool = True,
            **kwargs,
        ):
            if (
                isinstance(text, str)
                and len(text) >= min_chars
                and text.count(delim) >= 1
            ):
                return segment_batch_encode_ids(
                    self,
                    text,
                    split_delimiter=delim,
                    add_special_tokens=add_special_tokens,
                    segment_cache=cache,
                )
            return super().encode(
                text, *args, add_special_tokens=add_special_tokens, **kwargs
            )

        def __call__(self, text=None, text_pair=None, *args, **kwargs):  # type: ignore[override]
            # Single-string batch path used by TokenizerManager._tokenize_texts.
            if (
                text is not None
                and text_pair is None
                and isinstance(text, list)
                and len(text) == 1
                and isinstance(text[0], str)
                and len(text[0]) >= min_chars
                and text[0].count(delim) >= 1
                and not kwargs.get("return_token_type_ids")
            ):
                add_special = kwargs.get("add_special_tokens", True)
                return {
                    "input_ids": [
                        segment_batch_encode_ids(
                            self,
                            text[0],
                            split_delimiter=delim,
                            add_special_tokens=add_special,
                            segment_cache=cache,
                        )
                    ]
                }
            return super().__call__(text, text_pair, *args, **kwargs)

    SegmentBatchEncodeTokenizerImpl.__name__ = f"SegmentBatchEncode{base_cls.__name__}"
    tokenizer.__class__ = SegmentBatchEncodeTokenizerImpl
    logger.info(
        "segment_batch_encode enabled: delimiter=%r segment_cache=%s min_chars=%d",
        delim,
        segment_cache,
        min_chars,
    )
    return tokenizer
