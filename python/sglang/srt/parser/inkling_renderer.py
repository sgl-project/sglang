from __future__ import annotations

import functools
import json
import math
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
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
)

INKLING_DEFAULT_REASONING_EFFORT = 0.9
_THINKING_EFFORT_PREFIX = "Thinking effort level:"

TML_RENDERERS_INSTALL_HINT = (
    "Serving Inkling requires the `tml-renderers` package (TML's reference "
    "TMLv0 renderer/parser; wheels for Linux and macOS arm64). Install it with "
    "`pip install tml-renderers`."
)

_IMAGE_PART_TYPES = frozenset({"image", "input_image", "image_url"})
_AUDIO_PART_TYPES = frozenset({"audio", "input_audio", "audio_url"})
_TEXT_PART_TYPES = frozenset({None, "text", "input_text"})
_THINKING_PART_TYPES = frozenset({"thinking", "reasoning"})

_IMAGE_LOCATION_PREFIX = "sglang-inkling-image://"
_AUDIO_LOCATION_PREFIX = "sglang-inkling-audio://"

_NATIVE_TOKENIZERS = {
    "inkling": "O200kBaseChatTokenizer",
    "inkling2": "H200kBaseChatTokenizer",
}
# Inkling2 checkpoints keep these native audio IDs as unused reserved tokens.
_UNUSED_SPECIALS = {
    "inkling": (),
    "inkling2": ("content_audio_input", "audio", "audio_end"),
}


@functools.cache
def load_tml_renderers(spec: str = "inkling") -> SimpleNamespace:
    try:
        from tml_renderers import chat, tokenizers, v0
    except ImportError as exc:
        raise ImportError(TML_RENDERERS_INSTALL_HINT) from exc
    tokenizer = getattr(tokenizers, _NATIVE_TOKENIZERS[spec])()
    return SimpleNamespace(
        chat=chat,
        v0=v0,
        tokenizer=tokenizer,
        renderer=v0.Renderer(tokenizer),
        unused_special_ids=frozenset(
            tokenizer.encode_special(name) for name in _UNUSED_SPECIALS[spec]
        ),
    )


_TOKENIZER_PROBE = "Hello, 世界! def f(x): return x**2\n\t  naïve 👩‍🚀 12345678901"


def check_served_tokenizer(tokenizer: Any, spec: str) -> None:
    # Prompts and outputs use tml-renderers' tokenizer, while the scheduler matches
    # stop strings and detokenizes with the served one; they must agree.
    tml = load_tml_renderers(spec)
    # encode() is the lookup both the HF and the tiktoken tokenizer wrappers share.
    mismatched = []
    for token in INKLING_SPECIAL_TOKEN_IDS:
        token_id = tml.tokenizer.encode_special(token[2:-2])
        if token_id not in tml.unused_special_ids and tokenizer.encode(
            token, add_special_tokens=False
        ) != [token_id]:
            mismatched.append(token)
    if tokenizer.encode(_TOKENIZER_PROBE, add_special_tokens=False) != list(
        tml.tokenizer.encode_ordinary(_TOKENIZER_PROBE)
    ):
        mismatched.append("ordinary text")
    if mismatched:
        raise ValueError(
            "The served tokenizer disagrees with tml-renderers' "
            f"{type(tml.tokenizer).__name__} on {', '.join(mismatched)}; "
            "Inkling prompts and outputs would be misencoded."
        )


def render_inkling_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = "auto",
    parallel_tool_calls: bool = True,
    reasoning_effort: float | None = None,
    spec: str = "inkling",
) -> list[int]:
    """Media parts emit one placeholder each for later MM expansion.
    Inkling samples its own assistant header. The tool-choice instruction
    joins the first system message: the effort directive if none was sent."""
    tml = load_tml_renderers(spec)
    media_marker = f"sglang-inkling-media-{uuid.uuid4().hex}:"
    oss_messages, media_kinds = _to_oss_messages(messages, media_marker)
    tools = _tools_for_choice(tools, tool_choice)
    effort = (
        INKLING_DEFAULT_REASONING_EFFORT
        if reasoning_effort is None
        else reasoning_effort
    )
    oss_messages.insert(
        _leading_system_count(oss_messages), _thinking_effort_message(effort)
    )
    if tools:
        instruction = _tool_choice_instruction(tool_choice, parallel_tool_calls)
        if instruction is not None:
            oss_messages[0] = _with_appended_text(oss_messages[0], instruction)
        oss_messages.insert(0, _tool_declare_message(tools))
    turns = tml.chat.OpenAIMessage.from_oss_messages(oss_messages)
    for oss_message, turn in zip(oss_messages, turns, strict=True):
        # The library reads an author name only for tool messages.
        if oss_message["role"] == "assistant" and oss_message.get("name"):
            turn.author = tml.chat.Author(
                tml.chat.AuthorKind.Model, oss_message["name"]
            )
    native = [
        _resolve_media(message, tml.chat, media_marker, media_kinds)
        for turn in turns
        for message in turn.to_messages()
    ]
    spans, _parser = tml.renderer.render_for_completion(native)
    return _spans_to_input_ids(spans, tml)


def render_inkling_assistant_prefix(prefix: str, spec: str = "inkling") -> list[int]:
    """TMLv0 has no prefill; continuation opens a model text block without a terminator."""
    tokenizer = load_tml_renderers(spec).tokenizer
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
    call_names: dict[str, str] = {}
    for message in messages:
        oss_message = {
            key: value for key, value in message.items() if value is not None
        }
        if oss_message.get("role") == "developer":
            oss_message["role"] = "system"
        reasoning = oss_message.pop("reasoning_content", None)
        if reasoning and oss_message.get("role") == "assistant":
            oss_message["content"] = _with_leading_thinking(
                reasoning, oss_message.get("content")
            )
        for call in oss_message.get("tool_calls") or ():
            if call.get("id") and call["function"].get("name"):
                call_names[call["id"]] = call["function"]["name"]
        # The library takes a tool author only from ``name``; spec Chat clients and the
        # Responses/Anthropic adapters send just ``tool_call_id``.
        if oss_message.get("role") == "tool" and not oss_message.get("name"):
            name = call_names.get(oss_message.get("tool_call_id"))
            if name:
                oss_message["name"] = name
        content = oss_message.get("content")
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            oss_message["content"] = [
                _to_oss_part(part, media_marker, media_kinds) for part in content
            ]
        oss_messages.append(oss_message)
    return oss_messages, media_kinds


def _with_leading_thinking(reasoning: str, content: Any) -> list[Any]:
    # tml-renderers reads prior reasoning only from thinking content parts.
    parts: list[Any] = [{"type": "thinking", "thinking": reasoning}]
    if isinstance(content, str):
        if content:
            parts.append({"type": "text", "text": content})
    elif content:
        if any(
            isinstance(part, Mapping) and part.get("type") in _THINKING_PART_TYPES
            for part in content
        ):
            raise ValueError(
                "assistant message cannot mix reasoning_content with ordered "
                "thinking parts"
            )
        parts.extend(content)
    return parts


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


def _thinking_effort_message(effort: float) -> dict[str, Any]:
    if not math.isfinite(effort) or not 0.0 <= effort <= 1.0:
        raise ValueError(f"Inkling reasoning effort must be in [0, 1], got {effort}")
    return {
        "role": "system",
        "content": f"{_THINKING_EFFORT_PREFIX} {round(effort, 2):g}",
    }


def _tools_for_choice(
    tools: Sequence[Mapping[str, Any]] | None,
    tool_choice: str | Mapping[str, Any] | None,
) -> Sequence[Mapping[str, Any]] | None:
    if not tools or tool_choice == "none":
        return None
    name = _chosen_function_name(tool_choice)
    if name is None:
        return tools
    chosen = [
        tool for tool in tools if (tool.get("function") or {}).get("name") == name
    ]
    if not chosen:
        raise ValueError(f"tool_choice names an undeclared function: {name!r}")
    return chosen


def _tool_choice_instruction(
    tool_choice: str | Mapping[str, Any] | None, parallel_tool_calls: bool
) -> str | None:
    name = _chosen_function_name(tool_choice)
    if name is not None:
        target = f"the function named {json.dumps(name)}"
        if parallel_tool_calls:
            return f"Call {target} in your next response."
        return f"Call {target} exactly once in your next response."
    if tool_choice == "required":
        quantity = "at least one" if parallel_tool_calls else "exactly one"
        return f"Call {quantity} of the provided functions in your next response."
    if tool_choice not in (None, "auto"):
        raise ValueError(f"unsupported tool_choice: {tool_choice!r}")
    if parallel_tool_calls:
        return None
    return "If you call a function, call at most one function in your next response."


def _chosen_function_name(tool_choice: str | Mapping[str, Any] | None) -> str | None:
    if not isinstance(tool_choice, Mapping):
        return None
    name = (tool_choice.get("function") or {}).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"named tool_choice needs a function name: {tool_choice!r}")
    return name


def _with_appended_text(message: Mapping[str, Any], text: str) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str) and content:
        content = f"{content}\n\n{text}"
    elif content is None or isinstance(content, str):
        content = text
    else:
        content = [*content, {"type": "text", "text": text}]
    return {**message, "content": content}


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
            _append_media_placeholder(input_ids, span.location, tml)
        else:
            raise ValueError(
                f"unexpected TMLv0 span in Inkling prompt: {type(span).__name__}"
            )
    return input_ids


def _append_media_placeholder(
    input_ids: list[int], location: str, tml: SimpleNamespace
) -> None:
    if location.startswith(_IMAGE_LOCATION_PREFIX):
        input_ids.append(IMAGE_TOKEN_ID)
        return
    tokenizer = tml.tokenizer
    audio_input, audio_end = (
        tokenizer.encode_special(_special_name(token))
        for token in (CONTENT_AUDIO_INPUT, AUDIO_END)
    )
    if {audio_input, audio_end} & tml.unused_special_ids:
        raise ValueError("This Inkling checkpoint does not accept audio input")
    # tml-renderers DMel-encodes audio at render time, so audio rides in an image
    # slot and gets the MM processor's audio framing here.
    if input_ids[-1] != tokenizer.encode_special(_special_name(CONTENT_IMAGE)):
        raise ValueError("Inkling audio placeholder is not framed as a content block")
    input_ids[-1] = audio_input
    input_ids.extend([AUDIO_TOKEN_ID, audio_end])


def _special_name(token: str) -> str:
    return token.removeprefix("<|").removesuffix("|>")
