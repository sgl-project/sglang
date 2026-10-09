from __future__ import annotations

import functools
import json
import uuid
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
    AUDIO_TOKEN_ID,
    CONTENT_AUDIO_INPUT,
    CONTENT_IMAGE,
    CONTENT_TEXT,
    IMAGE_TOKEN_ID,
    MESSAGE_MODEL,
)

INKLING_DEFAULT_REASONING_EFFORT = 0.9

TML_RENDERERS_INSTALL_HINT = (
    "Serving Inkling requires the `tml-renderers` package (TML's reference "
    "TMLv0 renderer/parser, Python >= 3.11). Install it with "
    "`pip install tml-renderers`."
)

_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio", "audio_url"})
_TEXT_PART_TYPES = frozenset({None, "text", "input_text"})
_THINKING_PART_TYPES = frozenset({"thinking", "reasoning"})

_IMAGE_LOCATION_PREFIX = "sglang-inkling-image://"
_AUDIO_LOCATION_PREFIX = "sglang-inkling-audio://"


@functools.cache
def load_tml_renderers() -> SimpleNamespace:
    try:
        from tml_renderers import chat, tokenizers, v0
    except ImportError as exc:
        raise ImportError(TML_RENDERERS_INSTALL_HINT) from exc
    tokenizer = tokenizers.o200k_base_chat()
    return SimpleNamespace(
        chat=chat, v0=v0, tokenizer=tokenizer, renderer=v0.Renderer(tokenizer)
    )


def render_inkling_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    reasoning_effort: float | None = None,
) -> list[int]:
    """Render chat messages to Inkling input_ids via ``tml_renderers.v0``.

    Emits ONE ``IMAGE_TOKEN_ID`` / ``AUDIO_TOKEN_ID`` per media part; the MM
    processor expands them later. No assistant turn opener is appended: Inkling
    samples its own ``<|message_model|>`` header.
    """
    tml = load_tml_renderers()
    media_marker = f"sglang-inkling-media-{uuid.uuid4().hex}:"
    oss_messages, media_kinds = _to_oss_messages(messages, media_marker)
    if tools:
        oss_messages.insert(
            _leading_system_count(oss_messages), _tool_declare_message(tools)
        )
    native = [
        _resolve_media(message, tml.chat, media_marker, media_kinds)
        for oss_message in tml.chat.OpenAIMessage.from_oss_messages(oss_messages)
        for message in oss_message.to_messages()
    ]
    effort = (
        INKLING_DEFAULT_REASONING_EFFORT
        if reasoning_effort is None
        else reasoning_effort
    )
    spans, _parser = tml.renderer.render_for_completion_with_effort(native, effort)
    return _spans_to_input_ids(spans, tml)


def render_inkling_assistant_prefix(prefix: str) -> list[int]:
    """Open (unterminated) model text block that the model continues; TMLv0
    itself has no prefill, so this is an sglang ``continue_final_message``
    extension."""
    tokenizer = load_tml_renderers().tokenizer
    return [
        tokenizer.encode_special(_special_name(MESSAGE_MODEL)),
        tokenizer.encode_special(_special_name(CONTENT_TEXT)),
        *tokenizer.encode_ordinary(prefix),
    ]


def _to_oss_messages(
    messages: Sequence[Mapping[str, Any]], media_marker: str
) -> tuple[list[dict[str, Any]], list[str]]:
    media_kinds: list[str] = []
    oss_messages = []
    for message in messages:
        oss_message = {
            key: value for key, value in message.items() if value is not None
        }
        if oss_message.get("role") == "developer":
            oss_message["role"] = "system"
        content = oss_message.get("content")
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            oss_message["content"] = [
                _to_oss_part(part, media_marker, media_kinds) for part in content
            ]
        oss_messages.append(oss_message)
    return oss_messages, media_kinds


def _to_oss_part(
    part: Any, media_marker: str, media_kinds: list[str]
) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, Mapping):
        raise TypeError(f"content part must be mapping, got {type(part).__name__}")
    part_type = part.get("type")
    if part_type in _TEXT_PART_TYPES:
        return {"type": "text", "text": part.get("text") or ""}
    if part_type in _THINKING_PART_TYPES:
        text = part.get("thinking")
        if text is None:
            text = part.get("text", "")
        if not isinstance(text, str):
            raise TypeError("Inkling thinking part payload must be a string")
        return {"type": "thinking", "thinking": text}
    if part_type in _IMAGE_PART_TYPES or part_type in _AUDIO_PART_TYPES:
        media_kinds.append("image" if part_type in _IMAGE_PART_TYPES else "audio")
        return {"type": "text", "text": f"{media_marker}{len(media_kinds) - 1}"}
    raise ValueError(f"unsupported content part type: {part_type!r}")


def _resolve_media(message: Any, chat: Any, media_marker: str, media_kinds: list[str]):
    # Media bytes are encoded later by the MM processor, so each media part
    # travels through the renderer as a pointer whose span becomes ONE placeholder.
    content = message.content
    if not isinstance(content, chat.Text) or not content.text.startswith(media_marker):
        return message
    index = int(content.text.removeprefix(media_marker))
    if media_kinds[index] == "audio" and message.author.kind != chat.AuthorKind.User:
        raise ValueError("Inkling audio input is only supported in user messages")
    prefix = (
        _IMAGE_LOCATION_PREFIX
        if media_kinds[index] == "image"
        else _AUDIO_LOCATION_PREFIX
    )
    pointer = chat.ImagePointer(f"{prefix}{index}", chat.ImageFormat.Png, 1, 1)
    return message.copy(content=pointer)


def _leading_system_count(messages: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    while count < len(messages) and messages[count].get("role") == "system":
        count += 1
    return count


def _tool_declare_message(tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    declared = []
    for tool in tools:
        function = tool.get("function") or {}
        declared.append(
            {
                "type": tool.get("type") or "function",
                "function": {
                    key: value for key, value in function.items() if value is not None
                },
            }
        )
    return {
        "role": "tool_declare",
        "content": json.dumps(declared, ensure_ascii=False, separators=(",", ":")),
    }


def _spans_to_input_ids(spans: Sequence[Any], tml: SimpleNamespace) -> list[int]:
    input_ids: list[int] = []
    for wrapped in spans:
        span = wrapped.span
        if isinstance(span, tml.chat.EncodedTextTokenSpan):
            input_ids.extend(span.tokens)
        elif isinstance(span, tml.chat.ImageAssetPointerTokenSpan):
            _append_media_placeholder(input_ids, span.location, tml.tokenizer)
        else:
            raise ValueError(
                f"unexpected TMLv0 span in Inkling prompt: {type(span).__name__}"
            )
    return input_ids


def _append_media_placeholder(
    input_ids: list[int], location: str, tokenizer: Any
) -> None:
    if location.startswith(_IMAGE_LOCATION_PREFIX):
        input_ids.append(IMAGE_TOKEN_ID)
        return
    # tml-renderers DMel-encodes audio at render time, so audio rides in an image
    # slot and gets the MM processor's audio framing here.
    if input_ids[-1] != tokenizer.encode_special(_special_name(CONTENT_IMAGE)):
        raise ValueError("Inkling audio placeholder is not framed as a content block")
    input_ids[-1] = tokenizer.encode_special(_special_name(CONTENT_AUDIO_INPUT))
    input_ids.extend(
        [AUDIO_TOKEN_ID, tokenizer.encode_special(_special_name(AUDIO_END))]
    )


def _special_name(token: str) -> str:
    return token.removeprefix("<|").removesuffix("|>")
