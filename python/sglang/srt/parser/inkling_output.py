from __future__ import annotations

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
    """Incremental TMLv0 parse of one choice's sampled token IDs.

    Parsing runs on token IDs through ``tml_renderers.v0.Parser``, so a control
    token is recognized only by ID: model text that spells a marker stays text.
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
        self._text_opener_ids = (
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
        )
        self._end_message_id = tokenizer.encode_special("end_message")
        self._message_ids: list[int] = []
        self._streamed = ""
        self._held_reasoning = ""
        self._model_authored = True
        self._at_message_boundary = True
        self._in_unframed_text = False
        self._num_tool_calls = 0
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
        tokenizer = self._tml.tokenizer
        if (
            isinstance(matched_stop, int)
            and not keep_matched_stop
            and token_ids
            and token_ids[-1] == matched_stop
            and not tokenizer.is_special_token(matched_stop)
        ):
            token_ids = token_ids[:-1]
        delta = self.feed(token_ids).merge(self._flush())
        if isinstance(matched_stop, str):
            delta = _trim_stop_string(delta, matched_stop, keep_stop=keep_matched_stop)
        return delta

    def _flush(self) -> InklingOutputDelta:
        delta = _DeltaBuilder()
        try:
            updates = self._parser.flush_updates()
        except self._tml.v0.ParseError as exc:
            self._recover_message(delta, exc)
            return delta.build()
        for update in updates:
            self._apply(update.update, delta)
        self._release_held_reasoning(delta)
        return delta.build()

    def _feed_token(self, token_id: int, delta: _DeltaBuilder) -> None:
        is_special = self._tml.tokenizer.is_special_token(token_id)
        if self._in_unframed_text and is_special:
            # Inside a real text block TML reads most specials as text; the
            # sampler never emitted an opener here, so a special is framing.
            self._in_unframed_text = False
            if token_id != self._end_message_id:
                self._parse(self._end_message_id, delta)
            if token_id in self._content_kind_ids:
                return
        elif self._at_message_boundary and not is_special:
            # Constrained decoding (e.g. response_format) samples a bare payload
            # where a message header belongs; parse it as model text.
            for opener_id in self._text_opener_ids:
                self._parse(opener_id, delta)
            self._in_unframed_text = True
        self._parse(token_id, delta)

    def _parse(self, token_id: int, delta: _DeltaBuilder) -> None:
        self._at_message_boundary = False
        self._message_ids.append(token_id)
        try:
            updates = self._parser.parse_token(token_id)
        except self._tml.v0.ParseError as exc:
            self._recover_message(delta, exc)
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
                self._emit_text(update.content, update.content.text, delta)
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
                self._emit_text(content, content.text[len(self._streamed) :], delta)
            self._release_held_reasoning(delta)
        elif isinstance(content, chat.InvokeTool):
            if self._parse_tool_calls and content.text is None:
                delta.tool_calls.append(self._tool_call(message))
                self._num_tool_calls += 1
            else:
                delta.content.append(self._payload_text())

    def _emit_text(self, content: Any, text: str, delta: _DeltaBuilder) -> None:
        if not (
            self._separate_reasoning and isinstance(content, self._tml.chat.Thinking)
        ):
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
        return InklingToolCall(
            index=self._num_tool_calls,
            name=function["name"],
            arguments=function["arguments"],
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
        self._parser.reset()
        self._message_ids = []
        self._streamed = ""
        self._model_authored = True
        self._at_message_boundary = True
        self._in_unframed_text = False

    def _payload_text(self) -> str:
        ids = self._message_ids
        kind_index = next(
            (i for i, token_id in enumerate(ids) if token_id in self._content_kind_ids),
            -1,
        )
        tokenizer = self._tml.tokenizer
        return tokenizer.decode(
            [t for t in ids[kind_index + 1 :] if not tokenizer.is_special_token(t)]
        )


def _trim_stop_string(
    delta: InklingOutputDelta, stop: str, *, keep_stop: bool
) -> InklingOutputDelta:
    # The scheduler stops at the first occurrence; like the detokenizer's text
    # trim, drop it (unless kept) and whatever followed it in the last token.
    for field in ("reasoning", "content"):
        text = getattr(delta, field)
        pos = text.find(stop)
        if pos != -1:
            end = pos + len(stop) if keep_stop else pos
            return msgspec.structs.replace(delta, **{field: text[:end]})
    return delta


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
