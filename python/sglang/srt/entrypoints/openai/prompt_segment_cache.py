"""Encode a rendered chat prompt one message at a time and reuse the ids across turns.

An agent sends its whole history on every turn, so the chat endpoint encodes a
prompt that only grew by a few messages. The encode is a large CPU cost of a
long request and it runs on the server's event loop.

A HuggingFace fast tokenizer without a normalizer first cuts its added tokens
out of the text, leftmost and longest match first, and then encodes each piece
between them alone. The ids of the prompt are therefore the ids of its pieces,
joined, if every cut is a token boundary. This cache cuts the rendered prompt
in front of the special tokens that the chat template writes, keeps the ids of
each piece, and encodes only the pieces it has not seen. It uses a special
token as a cut point only if no added token can start before it and reach into
it.

The result is checked against a full encode on the first calls and then at a
fixed interval. On a mismatch the cache turns itself off.
"""

import logging
import re
from array import array
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# The scan for cut points tries each marker, so its cost grows with the count.
_MAX_MARKERS = 32
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
        # Longest first: the regex takes the first alternative that matches.
        self._markers = re.compile(
            "|".join(
                re.escape(marker) for marker in sorted(markers, key=len, reverse=True)
            )
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
        logger.info("Chat prompt segment cache: on, %d split markers", len(markers))
        return cls(tokenizer, markers, **kwargs)

    def _split(self, text: str) -> Iterator[str]:
        start = 0
        for match in self._markers.finditer(text):
            if match.start() > start:
                yield text[start : match.start()]
                start = match.start()
        if start < len(text):
            yield text[start:]

    def encode(self, text: str, encode_kwargs: Dict[str, Any]) -> List[int]:
        if not self.enabled:
            return self._tokenizer.encode(text, **encode_kwargs)
        kwargs_key = tuple(sorted(encode_kwargs.items()))
        ids: List[int] = []
        for segment in self._split(text):
            key = (segment, kwargs_key)
            cached = self._segments.get(key)
            if cached is None:
                cached = array("i", self._tokenizer.encode(segment, **encode_kwargs))
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
    # The module docstring describes the HuggingFace `tokenizers` backend only.
    # Other backends and slow tokenizers keep the full encode.
    if not isinstance(backend, Tokenizer) or backend.normalizer is not None:
        return []
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return []
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    tokens = list(added.values())
    # A token that strips whitespace or matches whole words depends on its neighbors.
    if any(token.lstrip or token.rstrip or token.single_word for token in tokens):
        return []
    # The backend cuts these tokens out of the raw text before any other step.
    first_pass = [token.content for token in tokens if not token.normalized]
    markers = sorted(
        {
            token.content
            for token in tokens
            if token.special
            and not token.normalized
            and token.content
            and token.content in template
            and not any(_can_reach_into(other, token.content) for other in first_pass)
        }
    )
    if not markers or len(markers) > _MAX_MARKERS:
        return []
    return markers


def _can_reach_into(token: str, marker: str) -> bool:
    """Can a match of `token` start before a `marker` in the text and reach into it?

    The backend takes the leftmost match, so such a match wins and the marker is
    not a token boundary there. `token` can be the marker itself: ">>" in ">>>".
    """
    if marker in token[1:]:
        return True
    longest = min(len(token), len(marker)) - 1
    return any(token.endswith(marker[:size]) for size in range(1, longest + 1))
