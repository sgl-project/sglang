from __future__ import annotations

import functools
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
    """Control tokens are identified by ID; ordinary tokens spelling a marker remain text."""

    def __init__(
        self,
        *,
        separate_reasoning: bool,
        parse_tool_calls: bool,
        stream_reasoning: bool = True,
        continues_text_block: bool = False,
        spec: str = "inkling",
    ):
        self._tml = load_tml_renderers(spec)
        self._byte_ids = native_byte_ids(spec)
        self._separate_reasoning = separate_reasoning
        self._hold_reasoning = separate_reasoning and not stream_reasoning
        self._parse_tool_calls = parse_tool_calls
        _spans, self._parser = self._tml.renderer.render_for_completion([])
        self._stream = self._parser.parse_stream()
        tokenizer = self._tml.tokenizer
        self._content_kind_ids = frozenset(
            tokenizer.encode_special(name) for name in _CONTENT_KIND_NAMES
        )
        self._text_opener_ids = (
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
        )
        self._end_message_id = tokenizer.encode_special("end_message")
        self._thinking_id = tokenizer.encode_special("content_thinking")
        self._message_ids: list[int] = []
        self._streamed = ""
        self._held_reasoning = ""
        self._held_recovered = ""
        self._model_authored = True
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
            self._feed_token(token_id, delta)
        self.num_consumed_tokens += len(token_ids)
        return delta.build()

    def abort(self, token_ids: Sequence[int] = ()) -> InklingOutputDelta:
        """Parses tokens generated before an abort without finalizing, so an open
        call is never completed into an executable one."""
        if self._finished:
            return InklingOutputDelta()
        self._finished = True
        return self.feed(token_ids)

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
        num_consumed_tokens = self.num_consumed_tokens + len(token_ids)
        partial = b""
        if isinstance(matched_stop, str):
            token_ids, partial = self._cut_at_stop_string(
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
        delta = _DeltaBuilder()
        for token_id in [*token_ids, *(self._byte_ids[byte] for byte in partial)]:
            self._feed_token(token_id, delta)
        self.num_consumed_tokens = num_consumed_tokens
        return delta.build().merge(self._flush())

    def _cut_at_stop_string(
        self, token_ids: Sequence[int], stop: str, *, keep_stop: bool
    ) -> tuple[list[int], bytes]:
        """Returns the whole tokens kept plus the kept byte prefix of the token the
        cut falls in; those bytes may continue a character from earlier tokens."""
        # The scheduler matched ``stop`` on the decoded ids, framing included,
        # and ended on the token completing it; cut the raw stream there.
        tokenizer = self._tml.tokenizer
        if not token_ids:
            return [], b""
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
            return list(token_ids), b""
        if keep_stop:
            cut += len(stop_bytes)
        kept = list(token_ids[:start])
        offset = 0
        for token_id, piece in zip(token_ids[start:], pieces):
            if offset + len(piece) <= cut:
                kept.append(token_id)
            elif offset < cut and not tokenizer.is_special_token(token_id):
                return kept, piece[: cut - offset]
            else:
                break
            offset += len(piece)
        return kept, b""

    def _flush(self) -> InklingOutputDelta:
        delta = _DeltaBuilder()
        if self._content_kind_id() is not None:
            # A block cut off (e.g. by max_tokens) is closed so the native parser
            # completes it as its own kind or rejects an incomplete payload.
            self._parse(self._end_message_id, delta)
        # A header with no content block yet carries no payload; native finish
        # would reject it as unterminated.
        if not self._message_ids:
            try:
                updates = self._stream.finish()
            except self._tml.v0.ParseError as exc:
                self._recover_message(delta, exc)
                return delta.build()
            for update in updates:
                self._apply(update.update, delta)
        self._release_held_reasoning(delta)
        return delta.build()

    def _content_kind_id(self) -> int | None:
        return next(
            (
                token_id
                for token_id in self._message_ids
                if token_id in self._content_kind_ids
            ),
            None,
        )

    def _feed_token(self, token_id: int, delta: _DeltaBuilder) -> None:
        if token_id in self._tml.unused_special_ids:
            raise ValueError(
                f"Generated token {token_id} is a reserved token of this checkpoint"
            )
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
        self._at_message_boundary = False
        self._message_ids.append(token_id)
        try:
            updates = self._stream.feed([token_id])
        except self._tml.v0.ParseError as exc:
            self._recover_message(
                delta, exc, block_open=token_id != self._end_message_id
            )
            return
        for update in updates:
            self._apply(update.update, delta)

    def _apply(self, update: Any, delta: _DeltaBuilder) -> None:
        chat = self._tml.chat
        if isinstance(update, chat.StreamingMessageHeader):
            self._model_authored = update.author.kind == chat.AuthorKind.Model
            self._streamed = ""
        elif isinstance(update, chat.StreamingContent):
            if self._model_authored:
                self._emit_text(
                    update.content.text,
                    delta,
                    thinking=isinstance(update.content, chat.Thinking),
                )
                self._streamed += update.content.text
        else:
            self._complete_message(update, delta)
            self._message_ids = []
            self._streamed = ""
            self._at_message_boundary = True

    def _complete_message(self, message: Any, delta: _DeltaBuilder) -> None:
        chat = self._tml.chat
        content = message.content
        if message.author.kind != chat.AuthorKind.Model:
            logger.debug("Dropping non-model Inkling output message: %r", message)
        elif isinstance(content, (chat.Text, chat.Thinking)):
            if content.text.startswith(self._streamed):
                self._emit_text(
                    content.text[len(self._streamed) :],
                    delta,
                    thinking=isinstance(content, chat.Thinking),
                )
            self._release_held_reasoning(delta)
        elif isinstance(content, chat.ToolError):
            delta.content.append(content.text)
        elif isinstance(content, chat.InvokeTool):
            if self._parse_tool_calls and content.text is None:
                delta.tool_calls.append(self._tool_call(message))
                self._num_tool_calls += 1
            else:
                delta.content.append(self._payload_text())

    def _emit_text(self, text: str, delta: _DeltaBuilder, *, thinking: bool) -> None:
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
        self._held_recovered = ""

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

    def _recover_message(
        self, delta: _DeltaBuilder, exc: Exception, *, block_open: bool = False
    ) -> None:
        logger.warning(
            "Inkling output failed TMLv0 parsing; surfacing as text: %s", exc
        )
        text = self._payload_text()
        kind_id = self._content_kind_id()
        reopen = (
            block_open
            and self._model_authored
            and kind_id in (self._thinking_id, self._text_opener_ids[1])
        )
        unframed = self._in_unframed_text
        if kind_id == self._thinking_id and self._hold_reasoning:
            # The decoded payload supersedes this segment's held text; segments
            # recovered earlier in the same block stay held until it closes.
            self._held_reasoning = self._held_recovered + text
            self._held_recovered = self._held_reasoning
            if not reopen:
                self._release_held_reasoning(delta)
        elif kind_id == self._thinking_id:
            if text.startswith(self._streamed):
                self._emit_text(text[len(self._streamed) :], delta, thinking=True)
        else:
            emitted = "" if self._held_reasoning else self._streamed
            if text.startswith(emitted):
                delta.content.append(text[len(emitted) :])
            self._held_reasoning = ""
            self._held_recovered = ""
        self._stream = self._parser.parse_stream()
        self._message_ids = []
        self._streamed = ""
        self._model_authored = True
        self._at_message_boundary = True
        self._in_unframed_text = False
        if reopen:
            # Native parsing rejects a block at its first invalid byte; reopen it so
            # the rest of the block keeps its channel until its own end_message.
            for token_id in (self._text_opener_ids[0], kind_id):
                self._parse(token_id, delta)
            self._in_unframed_text = unframed

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


@functools.cache
def native_byte_ids(spec: str = "inkling") -> tuple[int, ...]:
    tokenizer = load_tml_renderers(spec).tokenizer
    byte_ids: dict[int, int] = {}
    token_id = 0
    while len(byte_ids) < 256:
        if not tokenizer.is_special_token(token_id):
            data = tokenizer.decode_bytes([token_id])
            if len(data) == 1:
                byte_ids.setdefault(data[0], token_id)
        token_id += 1
    return tuple(byte_ids[byte] for byte in range(256))


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
