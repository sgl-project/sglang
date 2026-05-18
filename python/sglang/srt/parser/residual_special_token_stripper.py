# Copyright 2024-2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Residual special-token-string stripping for streaming output.

Some tokenizers (notably Qwen-family models) register chat-template role
delimiters such as ``<|im_start|>`` as added tokens with ``special=True``.
When the model emits the corresponding token ID, ``skip_special_tokens=True``
strips it during detokenization, as expected.

However, models occasionally hallucinate the *literal byte sequence* of a
role marker via BPE — typically during long reasoning content or under
unusual sampling conditions. In that case the detokenizer faithfully decodes
the bytes (because ``skip_special_tokens`` only filters by token ID), and
the marker leaks into the response stream. Downstream API consumers that
treat role markers as a turn-termination signal (Anthropic ``/v1/messages``,
some OpenAI tool-call routers) then abort the request mid-thought.

This module provides a streaming-safe stripper that removes these residual
marker strings before they leave the serving layer. Tokens consumed by the
reasoning parser (``<think>`` / ``</think>``) and tool-call parser
(``<tool_call>``, ``[TOOL_CALLS]``, …) are intentionally preserved so those
parsers can still detect them.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


# Marker strings that downstream parsers consume in-band. Stripping these
# would break reasoning extraction or tool-call routing, so they are excluded
# from the residual stripper by default. Keep this list narrow — only
# parser-consumed structural markers belong here.
_PARSER_CONSUMED_MARKERS: Set[str] = {
    # Reasoning parsers
    "<think>",
    "</think>",
    "[THINK]",
    "[/THINK]",
    # Tool-call parsers
    "<tool_call>",
    "</tool_call>",
    "[TOOL_CALLS]",
    "[/TOOL_CALLS]",
    "<|tool_call|>",
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
    "<|python_tag|>",
}


def get_residual_special_token_strings(
    tokenizer,
    extra_exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """Collect the set of special-token *strings* whose presence in
    client-visible output indicates a hallucinated role/structural marker.

    Args:
        tokenizer: A HuggingFace ``PreTrainedTokenizer``-compatible object.
            Must expose ``added_tokens_decoder`` (a ``Dict[int, AddedToken]``)
            or, as a fallback, ``all_special_tokens`` plus
            ``get_added_vocab()``.
        extra_exclude: Optional iterable of additional marker strings to
            preserve. Useful when a custom parser consumes a non-standard
            structural token in-band.

    Returns:
        Sorted list (descending by length) of marker strings, with parser-
        consumed markers excluded. Sorting by length ensures that the
        rolling-buffer stripper matches the longest applicable marker first
        when two markers share a prefix.
    """
    excluded: Set[str] = set(_PARSER_CONSUMED_MARKERS)
    if extra_exclude:
        excluded.update(extra_exclude)

    candidates: Set[str] = set()

    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if added_tokens_decoder:
        for token_id, added_token in added_tokens_decoder.items():
            try:
                is_special = bool(getattr(added_token, "special", False))
                content = getattr(added_token, "content", None) or str(added_token)
            except Exception:  # pragma: no cover — defensive
                continue
            if is_special and content:
                candidates.add(content)
    else:
        # Fallback for tokenizer wrappers that do not expose
        # added_tokens_decoder (older HF versions, custom shims).
        special_tokens = getattr(tokenizer, "all_special_tokens", []) or []
        candidates.update(str(t) for t in special_tokens if t)

    # Drop markers consumed by downstream parsers.
    markers = [m for m in candidates if m and m not in excluded]

    # Longest first so the matcher cannot half-strip a marker whose prefix
    # is itself another marker (rare in practice, but defends against
    # tokenizers that register both ``<|im_start|>`` and ``<|im_start|>user``).
    markers.sort(key=lambda s: (-len(s), s))
    return markers


class StreamingResidualStringStripper:
    """Incremental, streaming-safe stripper for residual marker strings.

    The stripper buffers the trailing bytes of each chunk that could be the
    start of a known marker, so a marker split across chunk boundaries is
    correctly removed rather than half-emitted.

    Usage::

        stripper = StreamingResidualStringStripper(markers)
        for chunk in stream:
            client.send(stripper.feed(chunk))
        client.send(stripper.flush())

    Notes:
        * ``markers`` may be empty, in which case ``feed`` is a passthrough.
        * The buffer is bounded by ``max_marker_len - 1`` characters; for a
          typical model this is well under 100 bytes of memory per stream.
        * Calling ``flush()`` is required at end-of-stream to release any
          buffered tail. Subsequent ``flush()`` calls return ``""``.
    """

    __slots__ = ("_markers", "_max_marker_len", "_buffer")

    def __init__(self, markers: Iterable[str]):
        self._markers: List[str] = [m for m in markers if m]
        self._max_marker_len: int = (
            max((len(m) for m in self._markers), default=0)
        )
        self._buffer: str = ""

    @property
    def active(self) -> bool:
        """True when at least one marker is registered."""
        return bool(self._markers)

    def feed(self, chunk: str) -> str:
        """Append ``chunk`` to the internal buffer and return the prefix
        that is safe to emit immediately (with complete markers removed)."""
        if not self._markers:
            return chunk
        if not chunk:
            return ""

        self._buffer += chunk

        # Determine how many trailing characters could still be the start
        # of a marker. Anything before that boundary can be safely flushed.
        hold_back = self._suffix_prefix_overlap(self._buffer)
        safe_end = len(self._buffer) - hold_back
        safe = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]

        return self._strip_complete_markers(safe)

    def flush(self) -> str:
        """Return any buffered tail with complete markers removed."""
        if not self._buffer:
            return ""
        tail = self._buffer
        self._buffer = ""
        return self._strip_complete_markers(tail)

    def _strip_complete_markers(self, text: str) -> str:
        if not text:
            return text
        for marker in self._markers:
            if marker in text:
                text = text.replace(marker, "")
        return text

    def _suffix_prefix_overlap(self, text: str) -> int:
        """Return the largest k such that ``text[-k:]`` is a proper prefix
        of at least one marker. ``k == 0`` means the buffer can be fully
        flushed."""
        if not text or self._max_marker_len <= 1:
            return 0
        max_check = min(len(text), self._max_marker_len - 1)
        best = 0
        for marker in self._markers:
            limit = min(max_check, len(marker) - 1)
            if limit <= best:
                continue
            tail = text[-limit:]
            # Shrink ``tail`` until it is a prefix of ``marker``.
            for k in range(limit, best, -1):
                if marker.startswith(tail[-k:]):
                    best = k
                    break
        return best
