from __future__ import annotations

import codecs
import json
import logging
from collections.abc import Sequence
from typing import Any

import msgspec

from sglang.srt.parser.inkling_renderer import load_tml_renderers

logger = logging.getLogger(__name__)

_CONTENT_KIND_NAMES = (
    "content_text",
    "content_thinking",
    "content_xml",
    "content_invoke_tool_json",
    "content_invoke_tool_text",
    "content_tool_error",
    "content_image",
    "content_audio_input",
)


class InklingToolCall(msgspec.Struct, frozen=True):
    index: int
    name: str
    arguments: str


class InklingOutputDelta(msgspec.Struct, frozen=True):
    reasoning: str = ""
    content: str = ""
    tool_calls: tuple[InklingToolCall, ...] = ()

    def merge(self, other: InklingOutputDelta) -> InklingOutputDelta:
        return InklingOutputDelta(
            reasoning=self.reasoning + other.reasoning,
            content=self.content + other.content,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class InklingOutputParser:
    """Control tokens are identified by ID; ordinary tokens spelling a marker remain text.

    TMLv0's parser only parses whole token sequences, so each message is parsed
    once when it closes; text and thinking stream from their tokens meanwhile.
    """

    def __init__(
        self,
        *,
        separate_reasoning: bool,
        parse_tool_calls: bool,
        stream_reasoning: bool = True,
        continues_text_block: bool = False,
    ):
        self._tml = load_tml_renderers()
        self._separate_reasoning = separate_reasoning
        self._hold_reasoning = separate_reasoning and not stream_reasoning
        self._parse_tool_calls = parse_tool_calls
        _spans, self._parser = self._tml.renderer.render_for_completion([])
        tokenizer = self._tml.tokenizer
        self._content_kind_ids = frozenset(
            tokenizer.encode_special(name) for name in _CONTENT_KIND_NAMES
        )
        self._streamed_kind_ids = frozenset(
            tokenizer.encode_special(name)
            for name in ("content_text", "content_thinking")
        )
        self._thinking_id = tokenizer.encode_special("content_thinking")
        self._header_ids = {
            tokenizer.encode_special(f"message_{role}"): role == "model"
            for role in ("model", "user", "system", "tool")
        }
        self._end_sampling_id = tokenizer.encode_special("content_model_end_sampling")
        # A parse that yields no content raises EmptyResponseError, so each closed
        # message is parsed after a non-empty one and that leading message dropped.
        self._parse_prefix = [
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
            *tokenizer.encode_ordinary("."),
            tokenizer.encode_special("end_message"),
        ]
        self._text_opener_ids = (
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
        )
        self._end_message_id = tokenizer.encode_special("end_message")
        self._framing_ids = frozenset(
            (*self._header_ids, *self._content_kind_ids, self._end_sampling_id)
        )
        self._message_ids: list[int] = []
        self._kind_id: int | None = None
        self._text_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._streamed = ""
        self._held_reasoning = ""
        self._model_authored = True
        self._awaiting_author = True
        self._at_message_boundary = True
        self._in_unframed_text = False
        self._num_tool_calls = 0
        self._finished = False
        self.num_consumed_tokens = 0
        if continues_text_block:
            # The prompt ended inside an open model text block, so the sampled
            # tokens carry no header; replay the opener so they parse as text.
            for token_id in self._text_opener_ids:
                self._parse(token_id, _DeltaBuilder())

    def feed(self, token_ids: Sequence[int]) -> InklingOutputDelta:
        delta = _DeltaBuilder()
        for token_id in token_ids:
            self.num_consumed_tokens += 1
            self._feed_token(token_id, delta)
        return delta.build()

    def finish(
        self,
        token_ids: Sequence[int] = (),
        *,
        matched_stop: int | str | None = None,
        keep_matched_stop: bool = False,
    ) -> InklingOutputDelta:
        # Callers see a sticky finish_reason on later chunks; flushing again would
        # re-emit an unclosed message's payload.
        if self._finished:
            return InklingOutputDelta()
        self._finished = True
        if isinstance(matched_stop, str):
            token_ids = self._cut_at_stop_string(
                token_ids, matched_stop, keep_stop=keep_matched_stop
            )
        elif (
            not keep_matched_stop
            and token_ids
            and token_ids[-1] == matched_stop
            and matched_stop != self._end_message_id
        ):
            # Inside an open block TML renders even a special stop as text;
            # only end_message closes a block without rendering.
            token_ids = token_ids[:-1]
        return self.feed(token_ids).merge(self._flush())

    def _cut_at_stop_string(
        self, token_ids: Sequence[int], stop: str, *, keep_stop: bool
    ) -> list[int]:
        # The scheduler matched ``stop`` on the decoded ids, framing included,
        # and ended on the token completing it; cut the raw stream there.
        tokenizer = self._tml.tokenizer
        if not token_ids:
            return []
        stop_bytes = stop.encode()
        pieces = [tokenizer.decode_bytes([token_ids[-1]])]
        last_len = region_len = len(pieces[0])
        start = len(token_ids) - 1
        # A match ending in the last token starts at most len(stop) - 1 bytes before it.
        while start > 0 and region_len < last_len + len(stop_bytes) - 1:
            start -= 1
            pieces.insert(0, tokenizer.decode_bytes([token_ids[start]]))
            region_len += len(pieces[0])
        cut = b"".join(pieces).find(stop_bytes)
        if cut == -1:
            return list(token_ids)
        if keep_stop:
            cut += len(stop_bytes)
        kept = list(token_ids[:start])
        offset = 0
        for token_id, piece in zip(token_ids[start:], pieces):
            if offset + len(piece) <= cut:
                kept.append(token_id)
            elif offset < cut and not tokenizer.is_special_token(token_id):
                partial = piece[: cut - offset].decode(errors="ignore")
                kept += tokenizer.encode_ordinary(partial)
            offset += len(piece)
        return kept

    def _flush(self) -> InklingOutputDelta:
        delta = _DeltaBuilder()
        self._surface_unclosed_message(delta)
        self._release_held_reasoning(delta)
        return delta.build()

    def _surface_unclosed_message(self, delta: _DeltaBuilder) -> None:
        # A message cut off (e.g. by max_tokens) never completes; only text kinds
        # streamed, so a call payload would otherwise vanish.
        if not self._model_authored or not any(
            token_id in self._content_kind_ids for token_id in self._message_ids
        ):
            return
        text = self._payload_text()
        if text.startswith(self._streamed):
            delta.content.append(text[len(self._streamed) :])

    def _feed_token(self, token_id: int, delta: _DeltaBuilder) -> None:
        is_special = self._tml.tokenizer.is_special_token(token_id)
        if self._in_unframed_text and is_special:
            # Inside a real text block TML reads most specials as text; the
            # sampler never emitted an opener here, so a special is framing.
            self._in_unframed_text = False
            if token_id != self._end_message_id:
                self._parse(self._end_message_id, delta)
        elif self._at_message_boundary and not is_special:
            # Headerless fallback output is model text; synthesize its missing opener.
            for opener_id in self._text_opener_ids:
                self._parse(opener_id, delta)
            self._in_unframed_text = True
        self._parse(token_id, delta)

    def _parse(self, token_id: int, delta: _DeltaBuilder) -> None:
        # TMLv0 skips every token, end_message included, before a message's header
        # or content kind; unframed output gets its opener in _feed_token.
        if self._awaiting_author and token_id in self._framing_ids:
            self._awaiting_author = False
            self._model_authored = self._header_ids.get(token_id, True)
            if token_id == self._end_sampling_id:
                self._reset_message()
                return
        self._at_message_boundary = False
        self._message_ids.append(token_id)
        if self._awaiting_author:
            return
        if token_id == self._end_message_id:
            self._close_message(delta)
        elif self._kind_id is None:
            if token_id in self._content_kind_ids:
                self._kind_id = token_id
        elif self._kind_id in self._streamed_kind_ids and self._model_authored:
            text = self._token_text(token_id)
            self._emit_text(self._kind_id == self._thinking_id, text, delta)
            self._streamed += text

    def _token_text(self, token_id: int) -> str:
        # Inside a text block a special reads as its spelling.
        tokenizer = self._tml.tokenizer
        if tokenizer.is_special_token(token_id):
            return self._text_decoder.decode(tokenizer.decode([token_id]).encode())
        return self._text_decoder.decode(tokenizer.decode_bytes([token_id]))

    def _close_message(self, delta: _DeltaBuilder) -> None:
        try:
            messages = self._parser.parse_tokens(
                self._parse_prefix + self._message_ids
            )[1:]
        except self._tml.v0.ParseError as exc:
            self._recover_message(delta, exc)
            return
        for message in messages:
            self._complete_message(message, delta)
        self._reset_message()

    def _reset_message(self) -> None:
        self._message_ids = []
        self._kind_id = None
        self._text_decoder.reset()
        self._streamed = ""
        self._awaiting_author = True
        self._at_message_boundary = True

    def _complete_message(self, message: Any, delta: _DeltaBuilder) -> None:
        chat = self._tml.chat
        content = message.content
        if message.author.kind != chat.AuthorKind.Model:
            logger.debug("Dropping non-model Inkling output message: %r", message)
        elif isinstance(content, (chat.Text, chat.Thinking)):
            if content.text.startswith(self._streamed):
                thinking = isinstance(content, chat.Thinking)
                self._emit_text(thinking, content.text[len(self._streamed) :], delta)
            self._release_held_reasoning(delta)
        elif isinstance(content, chat.InvokeTool):
            if self._parse_tool_calls and content.text is None:
                delta.tool_calls.append(self._tool_call(message))
                self._num_tool_calls += 1
            else:
                delta.content.append(self._payload_text())

    def _emit_text(self, thinking: bool, text: str, delta: _DeltaBuilder) -> None:
        if not (self._separate_reasoning and thinking):
            delta.content.append(text)
        elif self._hold_reasoning:
            self._held_reasoning += text
        else:
            delta.reasoning.append(text)

    def _release_held_reasoning(self, delta: _DeltaBuilder) -> None:
        if self._held_reasoning:
            delta.reasoning.append(self._held_reasoning)
            self._held_reasoning = ""

    def _tool_call(self, message: Any) -> InklingToolCall:
        chat = self._tml.chat
        oss = chat.OpenAIMessage.to_oss_messages(
            chat.OpenAIMessage.from_messages([message])
        )
        function = oss[0]["tool_calls"][0]["function"]
        # The bridge re-serializes args with sorted keys, ASCII escapes and big ints
        # as floats; build them from the sampled payload instead.
        args = json.loads(self._payload_json())["args"]
        return InklingToolCall(
            index=self._num_tool_calls,
            name=function["name"],
            arguments=json.dumps(args, ensure_ascii=False),
        )

    def _recover_message(self, delta: _DeltaBuilder, exc: Exception) -> None:
        logger.warning(
            "Inkling output failed TMLv0 parsing; surfacing as text: %s", exc
        )
        text = self._payload_text()
        emitted = "" if self._held_reasoning else self._streamed
        if text.startswith(emitted):
            delta.content.append(text[len(emitted) :])
        self._held_reasoning = ""
        self._reset_message()
        self._model_authored = True
        self._in_unframed_text = False

    def _payload_ids(self) -> list[int]:
        ids = self._message_ids
        kind_index = next(
            (i for i, token_id in enumerate(ids) if token_id in self._content_kind_ids),
            -1,
        )
        return ids[kind_index + 1 :]

    def _payload_text(self) -> str:
        tokenizer = self._tml.tokenizer
        return tokenizer.decode(
            [t for t in self._payload_ids() if not tokenizer.is_special_token(t)]
        )

    def _payload_json(self) -> str:
        # A special inside the payload keeps its spelling, as in the library's parse.
        ids = self._payload_ids()
        if ids and ids[-1] == self._end_message_id:
            ids = ids[:-1]
        return self._tml.tokenizer.decode(ids)


class _DeltaBuilder:
    def __init__(self):
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.tool_calls: list[InklingToolCall] = []

    def build(self) -> InklingOutputDelta:
        return InklingOutputDelta(
            reasoning="".join(self.reasoning),
            content="".join(self.content),
            tool_calls=tuple(self.tool_calls),
        )
