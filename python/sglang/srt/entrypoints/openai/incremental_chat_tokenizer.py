"""Exact incremental tokenization of append-only chat prompts.

A multi-turn chat request re-sends the whole conversation, so the rendered
prompt of turn N+1 usually starts with the rendered history of turn N. Encoding
that prompt from scratch costs time linear in the history on every request, on
the tokenizer manager's event loop, ahead of prefill.

``IncrementalChatTokenizer`` reuses the token ids of the longest previously
seen history whose messages are a prefix of the current ones, and encodes only
the new text. The result is identical to ``tokenizer.encode(prompt)``: a suffix
is appended only when it begins with a special added token that strips no
whitespace, and the cached prefix does not end with a token that strips to its
right. Fast tokenizers split on such tokens before normalization and
pre-tokenization, so the encoding of the joined text is the concatenation of
the two encodings. Any other shape is encoded in full.

The generation prompt (the text a template appends for
``add_generation_prompt=True``) is learned once per template context from two
probe renders, so the history is taken from the prompt the server already
rendered rather than rendering the conversation a second time.

Enable with ``--enable-incremental-chat-tokenization``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import orjson
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 128
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_PROBE_MESSAGES = [{"role": "user", "content": "probe"}]


class _Entry(msgspec.Struct, frozen=True):
    message_digests: Tuple[bytes, ...]
    history_text: str
    history_ids: Tuple[int, ...]
    size_bytes: int


class IncrementalChatTokenizerStats(msgspec.Struct):
    calls: int = 0
    history_reuses: int = 0
    incremental_results: int = 0
    full_encodes: int = 0
    verification_failures: int = 0


def _digest(value: Any) -> bytes:
    return hashlib.sha256(
        orjson.dumps(value, option=orjson.OPT_SORT_KEYS, default=str)
    ).digest()


class IncrementalChatTokenizer:
    """Bounded LRU of rendered chat histories and their token ids."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        verify_every: int = 0,
    ) -> None:
        if max_entries < 1 or max_bytes < 1 or verify_every < 0:
            raise ValueError("invalid incremental chat tokenizer bounds")
        self.tokenizer = tokenizer
        # The HF property rebuilds this dict on every access; added tokens are fixed.
        self._added_tokens = dict(tokenizer.added_tokens_decoder)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.verify_every = verify_every
        self.stats = IncrementalChatTokenizerStats()
        # (context digest, message digests) -> entry, least recently used first
        self._entries: OrderedDict[Tuple[bytes, Tuple[bytes, ...]], _Entry] = (
            OrderedDict()
        )
        self._bytes = 0
        self._generation_suffix: Dict[bytes, Optional[Tuple[str, Tuple[int, ...]]]] = {}

    @staticmethod
    def supports(tokenizer: Any) -> bool:
        """Only fast (Rust) tokenizers split on added tokens before encoding."""
        return isinstance(tokenizer, PreTrainedTokenizerFast)

    def encode(
        self,
        messages: Sequence[Dict[str, Any]],
        prompt_text: str,
        *,
        tools: Optional[List[Dict]],
        template_kwargs: Dict[str, Any],
        encode_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Return ``tokenizer.encode(prompt_text, **encode_kwargs)``.

        ``prompt_text`` must be the render of ``messages`` with
        ``add_generation_prompt=True`` and the given tools and kwargs.
        """
        self.stats.calls += 1
        context = _digest(
            (self.tokenizer.chat_template, tools, template_kwargs, encode_kwargs)
        )
        message_digests = tuple(_digest(message) for message in messages)
        generation = self._learn_generation_suffix(
            context=context,
            tools=tools,
            template_kwargs=template_kwargs,
            encode_kwargs=encode_kwargs,
        )
        history_text = self._history_text(
            messages=messages,
            prompt_text=prompt_text,
            generation=generation,
            tools=tools,
            template_kwargs=template_kwargs,
        )
        history_ids, reused = self._encode_history(
            context=context,
            message_digests=message_digests,
            history_text=history_text,
            encode_kwargs=encode_kwargs,
        )
        prompt_ids, appended = self._encode_prompt(
            prompt_text=prompt_text,
            history_text=history_text,
            history_ids=history_ids,
            generation=generation,
            encode_kwargs=encode_kwargs,
        )
        self._store(context, message_digests, history_text, history_ids)
        if reused:
            self.stats.history_reuses += 1
        if reused or appended:
            self.stats.incremental_results += 1
            return self._maybe_verify(prompt_ids, prompt_text, encode_kwargs)
        return prompt_ids

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    # Internals

    def _history_text(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        prompt_text: str,
        generation: Optional[Tuple[str, Tuple[int, ...]]],
        tools: Optional[List[Dict]],
        template_kwargs: Dict[str, Any],
    ) -> str:
        if generation is not None and prompt_text.endswith(generation[0]):
            return prompt_text[: len(prompt_text) - len(generation[0])]
        return self._render(
            messages, tools, template_kwargs, add_generation_prompt=False
        )

    def _encode_history(
        self,
        *,
        context: bytes,
        message_digests: Tuple[bytes, ...],
        history_text: str,
        encode_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], bool]:
        entry = self._longest_prefix(context, message_digests)
        if entry is not None and history_text.startswith(entry.history_text):
            ids = self._append(
                entry.history_ids,
                history_text[len(entry.history_text) :],
                encode_kwargs,
            )
            if ids is not None:
                return ids, True
        self.stats.full_encodes += 1
        return self._encode(history_text, encode_kwargs), False

    def _encode_prompt(
        self,
        *,
        prompt_text: str,
        history_text: str,
        history_ids: List[int],
        generation: Optional[Tuple[str, Tuple[int, ...]]],
        encode_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], bool]:
        if prompt_text == history_text:
            return history_ids, False
        if (
            generation is not None
            and prompt_text == history_text + generation[0]
            and self._ends_at_safe_boundary(history_ids)
        ):
            return history_ids + list(generation[1]), True
        if prompt_text.startswith(history_text):
            ids = self._append(
                history_ids, prompt_text[len(history_text) :], encode_kwargs
            )
            if ids is not None:
                return ids, True
        self.stats.full_encodes += 1
        return self._encode(prompt_text, encode_kwargs), False

    def _maybe_verify(
        self, prompt_ids: List[int], prompt_text: str, encode_kwargs: Dict[str, Any]
    ) -> List[int]:
        if not self.verify_every or self.stats.incremental_results % self.verify_every:
            return prompt_ids
        reference = self._encode(prompt_text, encode_kwargs)
        if reference != prompt_ids:
            self.stats.verification_failures += 1
            self.clear()
            logger.error(
                "Incremental chat tokenization diverged from a full encode; cache cleared"
            )
        return reference

    def _render(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[Dict]],
        template_kwargs: Dict[str, Any],
        *,
        add_generation_prompt: bool,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=False,
            **template_kwargs,
        )
        if not isinstance(text, str):
            raise TypeError("chat template did not render a string")
        return text

    def _encode(self, text: str, encode_kwargs: Dict[str, Any]) -> List[int]:
        return list(self.tokenizer.encode(text, **encode_kwargs))

    def _starts_at_safe_boundary(self, token_id: int) -> bool:
        token = self._added_tokens.get(token_id)
        return (
            token is not None
            and token.special
            and not token.lstrip
            and not token.rstrip
        )

    def _ends_at_safe_boundary(self, prefix_ids: Sequence[int]) -> bool:
        if not prefix_ids:
            return True
        token = self._added_tokens.get(prefix_ids[-1])
        return token is None or not token.rstrip

    def _append(
        self,
        prefix_ids: Sequence[int],
        suffix_text: str,
        encode_kwargs: Dict[str, Any],
    ) -> Optional[List[int]]:
        if not suffix_text:
            return list(prefix_ids)
        suffix_ids = self._encode(suffix_text, encode_kwargs)
        if not suffix_ids:
            return list(prefix_ids)
        if not (
            self._starts_at_safe_boundary(suffix_ids[0])
            and self._ends_at_safe_boundary(prefix_ids)
        ):
            return None
        return [*prefix_ids, *suffix_ids]

    def _learn_generation_suffix(
        self,
        *,
        context: bytes,
        tools: Optional[List[Dict]],
        template_kwargs: Dict[str, Any],
        encode_kwargs: Dict[str, Any],
    ) -> Optional[Tuple[str, Tuple[int, ...]]]:
        if context in self._generation_suffix:
            return self._generation_suffix[context]
        result = None
        try:
            with_prompt = self._render(
                _PROBE_MESSAGES, tools, template_kwargs, add_generation_prompt=True
            )
            without = self._render(
                _PROBE_MESSAGES, tools, template_kwargs, add_generation_prompt=False
            )
            if with_prompt.startswith(without) and len(with_prompt) > len(without):
                suffix = with_prompt[len(without) :]
                suffix_ids = self._encode(suffix, encode_kwargs)
                if suffix_ids and self._starts_at_safe_boundary(suffix_ids[0]):
                    result = (suffix, tuple(suffix_ids))
        except Exception:  # a template that rejects the probe renders history instead
            logger.debug("generation-prompt probe failed", exc_info=True)
        self._generation_suffix[context] = result
        return result

    def _longest_prefix(
        self, context: bytes, message_digests: Tuple[bytes, ...]
    ) -> Optional[_Entry]:
        best_key = None
        best: Optional[_Entry] = None
        for key, entry in self._entries.items():
            n = len(entry.message_digests)
            if (
                key[0] == context
                and n <= len(message_digests)
                and (best is None or n > len(best.message_digests))
                and entry.message_digests == message_digests[:n]
            ):
                best_key, best = key, entry
        if best_key is not None:
            self._entries.move_to_end(best_key)
        return best

    def _store(
        self,
        context: bytes,
        message_digests: Tuple[bytes, ...],
        history_text: str,
        history_ids: Sequence[int],
    ) -> None:
        size_bytes = len(history_text.encode("utf-8")) + 8 * len(history_ids)
        if size_bytes > self.max_bytes:
            return
        key = (context, message_digests)
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old.size_bytes
        self._entries[key] = _Entry(
            message_digests=message_digests,
            history_text=history_text,
            history_ids=tuple(history_ids),
            size_bytes=size_bytes,
        )
        self._bytes += size_bytes
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.size_bytes
