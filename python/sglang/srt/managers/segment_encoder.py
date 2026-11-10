"""Encode prompts piece by piece, reusing pieces seen in earlier requests.

A fast tokenizer cuts its input at verbatim (normalized=False) added tokens
before it normalizes or pre-tokenizes, so the text between two of them is
encoded on its own.
Encoding those pieces separately therefore gives the same ids as one full
encode, as long as nothing in the pipeline looks across a piece boundary.
Chat templates put added tokens around every message, and an agentic turn
resends the whole conversation, so all but the newest pieces are cache hits.
"""

from __future__ import annotations

import json
import logging
import re
from array import array
from collections import OrderedDict

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Pre-tokenizers that treat every piece the same way. Metaspace is left out:
# with prepend_scheme="first" it only touches the first piece.
_PIECEWISE_PRE_TOKENIZERS = frozenset(
    {
        "Sequence",
        "Split",
        "ByteLevel",
        "Digits",
        "Punctuation",
        "Whitespace",
        "WhitespaceSplit",
    }
)

# Unicode normal forms act on each character with its combining marks, and
# an added token always starts a new character.
_PIECEWISE_NORMALIZERS = frozenset({"Sequence", "NFC", "NFD", "NFKC", "NFKD"})

# Arbitrary; a cached token costs about 10 bytes (key text plus int32 id).
_MAX_CACHED_TOKENS = 1 << 21


def _pre_tokenizer_unsupported_reason(spec: dict) -> str | None:
    kind = spec["type"]
    if kind not in _PIECEWISE_PRE_TOKENIZERS:
        return f"pre-tokenizer {kind}"
    if kind == "ByteLevel" and spec["add_prefix_space"]:
        return "ByteLevel pre-tokenizer adds a prefix space"
    for child in spec.get("pretokenizers", ()):
        reason = _pre_tokenizer_unsupported_reason(child)
        if reason is not None:
            return reason
    return None


def _normalizer_unsupported_reason(spec: dict) -> str | None:
    if spec["type"] not in _PIECEWISE_NORMALIZERS:
        return f"normalizer {spec['type']}"
    for child in spec.get("normalizers", ()):
        reason = _normalizer_unsupported_reason(child)
        if reason is not None:
            return reason
    return None


def _unsupported_reason(tokenizer) -> str | None:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return f"{type(tokenizer).__name__} is not a fast tokenizer"
    backend = tokenizer.backend_tokenizer
    if backend.normalizer is not None:
        reason = _normalizer_unsupported_reason(
            json.loads(backend.normalizer.__getstate__())
        )
        if reason is not None:
            return reason
    if backend.encode_special_tokens:
        return "special tokens are encoded as text"
    if backend.pre_tokenizer is not None:
        reason = _pre_tokenizer_unsupported_reason(
            json.loads(backend.pre_tokenizer.__getstate__())
        )
        if reason is not None:
            return reason
    for token in backend.get_added_tokens_decoder().values():
        if token.lstrip or token.rstrip or token.single_word or token.normalized:
            return f"added token {token.content!r} is not matched verbatim"
    if tokenizer.encode("x") != tokenizer.encode("x", add_special_tokens=False):
        return "the tokenizer adds special tokens"
    return None


def _self_test_texts(added_tokens: list[str]) -> list[str]:
    # Added tokens at the edges, back to back, cut short, and next to
    # whitespace, digits, non-ASCII text and combining marks.
    texts = []
    for token in added_tokens[:64]:
        texts.append(f"{token}a {token}{token}\n\n  12345 x{token[:-1]} {token}")
        texts.append(f" \u00fcber{token}\t{token} {token[1:]}\u4f60\u597d {token}")
        texts.append(f"e{token}\u0301e\u0301 \ufb01{token}\u0301")
    return texts


class SegmentCachedEncoder:
    """Same ids as ``tokenizer.encode(text)``, with earlier pieces cached."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self._backend = tokenizer.backend_tokenizer
        self._added_ids = {
            token.content: token_id
            for token_id, token in self._backend.get_added_tokens_decoder().items()
        }
        # Longest first, so the alternation picks the longest token at each
        # position, as the tokenizer's own matcher does.
        alternatives = sorted(self._added_ids, key=len, reverse=True)
        self._splitter = re.compile(
            "(" + "|".join(re.escape(token) for token in alternatives) + ")"
        )
        self._cache: OrderedDict[str, array] = OrderedDict()
        self._cached_tokens = 0

    @classmethod
    def build(cls, tokenizer) -> SegmentCachedEncoder | None:
        """None when piecewise encoding cannot be shown to match a full encode."""
        reason = _unsupported_reason(tokenizer)
        if reason is not None:
            logger.info("Prompt segment cache disabled: %s", reason)
            return None
        encoder = cls(tokenizer)
        if not encoder._added_ids:
            logger.info("Prompt segment cache disabled: no added tokens")
            return None
        for text in _self_test_texts(list(encoder._added_ids)):
            if encoder.encode(text) != tokenizer.encode(text):
                logger.warning(
                    "Prompt segment cache disabled: piecewise encode of "
                    "%r differs from a full encode",
                    text,
                )
                return None
        encoder._cache.clear()
        encoder._cached_tokens = 0
        return encoder

    def encode(self, text: str) -> list[int]:
        # re.split with one group alternates text pieces and added tokens.
        pieces = self._splitter.split(text)
        self._encode_misses(pieces[0::2])
        ids: list[int] = []
        for index, piece in enumerate(pieces):
            if index % 2:
                ids.append(self._added_ids[piece])
            elif piece:
                ids.extend(self._cache[piece])
                self._cache.move_to_end(piece)
        self._evict()
        return ids

    def _encode_misses(self, text_pieces: list[str]) -> None:
        misses = list(
            dict.fromkeys(p for p in text_pieces if p and p not in self._cache)
        )
        if not misses:
            return
        # A padded call on the shared tokenizer (a multimodal processor's
        # padding=True) leaves padding on the backend; a plain encode clears it
        # first, and so must this, or short pieces come back padded.
        if self._backend.padding is not None:
            self._backend.no_padding()
        if self._backend.truncation is not None:
            self._backend.no_truncation()
        encodings = self._backend.encode_batch_fast(misses, add_special_tokens=False)
        for piece, encoding in zip(misses, encodings):
            ids = array("i", encoding.ids)
            self._cache[piece] = ids
            self._cached_tokens += len(ids)

    def _evict(self) -> None:
        while self._cached_tokens > _MAX_CACHED_TOKENS and len(self._cache) > 1:
            _, ids = self._cache.popitem(last=False)
            self._cached_tokens -= len(ids)
