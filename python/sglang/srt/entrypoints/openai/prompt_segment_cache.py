"""Encode a rendered chat prompt one message at a time and reuse the ids across turns.

An agent sends its whole history on every turn, so the chat endpoint encodes a
prompt that only grew by a few messages. The encode is the largest CPU cost of
a long request and it runs on the server's event loop: about 130 ms for a
110K-token prompt.

A fast tokenizer without a normalizer first takes its special tokens out of the
text and then encodes each piece between them alone. The ids of the prompt are
therefore the ids of its pieces, joined. This cache splits the rendered prompt
in front of the special tokens that the chat template writes, keeps the ids of
each piece, and encodes only the pieces it has not seen.

The result is checked against a full encode on the first calls and then at a
fixed interval. On a mismatch the cache turns itself off.
"""

import logging
import re
from array import array
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_MARKERS = 256
_DEFAULT_MAX_CHARS = 64 * 1024 * 1024
_DEFAULT_CHECK_FIRST = 3
_DEFAULT_CHECK_EVERY = 512


class PromptSegmentCache:
    def __init__(
        self,
        tokenizer,
        markers: List[str],
        *,
        max_chars: int = _DEFAULT_MAX_CHARS,
        check_first: int = _DEFAULT_CHECK_FIRST,
        check_every: int = _DEFAULT_CHECK_EVERY,
    ):
        self._tokenizer = tokenizer
        self._split = re.compile(
            "(?=" + "|".join(re.escape(marker) for marker in markers) + ")"
        )
        self._segments: OrderedDict[tuple, array] = OrderedDict()
        self._chars = 0
        self._max_chars = max_chars
        self._check_first = check_first
        self._check_every = check_every
        self._calls = 0
        self.enabled = True

    @classmethod
    def create(cls, tokenizer, **kwargs) -> Optional["PromptSegmentCache"]:
        """Return a cache when per-piece encoding is exact for this tokenizer."""
        markers = _split_markers(tokenizer)
        if not markers:
            return None
        return cls(tokenizer, markers, **kwargs)

    def encode(self, text: str, encode_kwargs: Dict[str, Any]) -> List[int]:
        kwargs_key = tuple(sorted(encode_kwargs.items()))
        ids: List[int] = []
        for segment in self._split.split(text):
            if not segment:
                continue
            key = (segment, kwargs_key)
            cached = self._segments.get(key)
            if cached is None:
                cached = array("q", self._tokenizer.encode(segment, **encode_kwargs))
                self._segments[key] = cached
                self._chars += len(segment)
            else:
                self._segments.move_to_end(key)
            ids.extend(cached)
        while self._chars > self._max_chars and len(self._segments) > 1:
            (evicted, _), _ = self._segments.popitem(last=False)
            self._chars -= len(evicted)

        self._calls += 1
        if self._calls <= self._check_first or self._calls % self._check_every == 0:
            full = self._tokenizer.encode(text, **encode_kwargs)
            if ids != full:
                logger.warning(
                    "Per-message prompt encoding differs from a full encode. "
                    "The prompt segment cache is now off."
                )
                self.enabled = False
                self._segments.clear()
                self._chars = 0
                return full
        return ids


def _split_markers(tokenizer) -> List[str]:
    """Special tokens of the chat template that are safe places to split the prompt."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or getattr(backend, "normalizer", None) is not None:
        return []
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return []
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    tokens = list(added.values())
    # A token that strips whitespace or matches whole words depends on its neighbors.
    if any(token.lstrip or token.rstrip or token.single_word for token in tokens):
        return []
    markers = sorted(
        {
            token.content
            for token in tokens
            if token.special and token.content and token.content in template
        }
    )
    if not markers or len(markers) > _MAX_MARKERS:
        return []
    return markers
