# SPDX-License-Identifier: Apache-2.0
"""Translations between Responses-API wire shapes and SGLang's chat internals.

Two things the Responses API expresses that the chat-completions path has no
representation for:

* ``custom`` tools, whose payload is freeform text rather than JSON-object
  arguments. Each is surfaced to the model as a function tool with a single
  string property, and the resulting call is translated back into a
  ``custom_tool_call``.
* ``reasoning.encrypted_content``, the opaque blob a ``store=false`` client
  replays to hand a reasoning trace back to the server.
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, Optional, Set, Tuple

CUSTOM_TOOL_INPUT_KEY = "input"

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def custom_tool_parameters() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            CUSTOM_TOOL_INPUT_KEY: {
                "type": "string",
                "description": "The freeform text payload for this tool.",
            }
        },
        "required": [CUSTOM_TOOL_INPUT_KEY],
        "additionalProperties": False,
    }


def custom_tool_description(
    description: Optional[str], tool_format: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Fold a custom tool's description and its input format into one blob.

    A ``grammar`` format is described to the model rather than enforced by the
    grammar backend: the payload is emitted inside the model's own tool-call
    envelope, so constrained decoding cannot be scoped to it.
    """
    parts = []
    if description:
        parts.append(description)
    if isinstance(tool_format, dict) and tool_format.get("type") == "grammar":
        syntax = tool_format.get("syntax") or "lark"
        definition = tool_format.get("definition") or ""
        parts.append(
            f"The payload must conform to this {syntax} grammar:\n{definition}"
        )
    return "\n\n".join(parts) or None


def custom_tool_names(tools: Any) -> Set[str]:
    return {tool.name for tool in tools or [] if tool.type == "custom" and tool.name}


def encode_custom_tool_input(payload: str) -> str:
    return json.dumps({CUSTOM_TOOL_INPUT_KEY: payload}, ensure_ascii=False)


def decode_custom_tool_input(arguments: str) -> str:
    """Recover the payload from a completed shim call's arguments.

    Falls back to the raw string so a model that ignored the wrapper and
    emitted bare text still produces a usable call.
    """
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict):
        value = parsed.get(CUSTOM_TOOL_INPUT_KEY)
        if isinstance(value, str):
            return value
        if len(parsed) == 1:
            only = next(iter(parsed.values()))
            if isinstance(only, str):
                return only
    return arguments


def _payload_start(buffer: str) -> Optional[int]:
    key = f'"{CUSTOM_TOOL_INPUT_KEY}"'
    key_at = buffer.find(key)
    if key_at < 0:
        return None
    colon = buffer.find(":", key_at + len(key))
    if colon < 0:
        return None
    quote = buffer.find('"', colon + 1)
    return quote + 1 if quote >= 0 else None


def _unicode_escape(buffer: str, i: int) -> Optional[Tuple[str, int]]:
    if i + 6 > len(buffer):
        return None
    try:
        code = int(buffer[i + 2 : i + 6], 16)
    except ValueError:
        return None
    if not 0xD800 <= code <= 0xDBFF:
        return chr(code), i + 6
    if i + 12 > len(buffer) or buffer[i + 6 : i + 8] != "\\u":
        return None
    try:
        low = int(buffer[i + 8 : i + 12], 16)
    except ValueError:
        return None
    if not 0xDC00 <= low <= 0xDFFF:
        return None
    return chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)), i + 12


def decode_custom_tool_input_prefix(buffer: str) -> str:
    """Longest fully decodable prefix of the payload in a partial argument buffer.

    Streaming tool-call parsers hand out argument JSON in fragments, so the
    payload is un-escaped incrementally to keep ``custom_tool_call_input``
    deltas in step with the value the item finally reports.
    """
    start = _payload_start(buffer)
    if start is None:
        return ""
    out = []
    i, n = start, len(buffer)
    while i < n:
        ch = buffer[i]
        if ch == '"':
            break
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            break
        simple = _SIMPLE_ESCAPES.get(buffer[i + 1])
        if simple is not None:
            out.append(simple)
            i += 2
            continue
        if buffer[i + 1] == "u":
            decoded = _unicode_escape(buffer, i)
            if decoded is None:
                break
            out.append(decoded[0])
            i = decoded[1]
            continue
        break
    return "".join(out)


DEVELOPER_BLOCK_LABEL = "Developer instructions:"


def label_developer_content(content: Any) -> Any:
    """Prefix a ``developer`` message's text with its tier label.

    Chat templates recognize system/user/assistant/tool, so a developer message
    collapses to ``system``. Labelling the block keeps the tier legible instead
    of flattening it into the surrounding instructions, where a model can read
    it as one more conversational turn and let a later user message win.
    """
    if isinstance(content, str):
        return f"{DEVELOPER_BLOCK_LABEL}\n{content}"
    if isinstance(content, list):
        for index, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                labelled = dict(part)
                labelled["text"] = f"{DEVELOPER_BLOCK_LABEL}\n{part['text']}"
                return [*content[:index], labelled, *content[index + 1 :]]
        return [{"type": "input_text", "text": DEVELOPER_BLOCK_LABEL}, *content]
    return content


_REASONING_STATE_PREFIX = "sglang-reasoning-v1."


def encode_reasoning_state(text: str) -> str:
    """Pack a reasoning trace into the blob ``reasoning.encrypted_content`` carries.

    A ``store=false`` client replays reasoning items verbatim, so the trace has
    to survive the round trip without server-side state. SGLang holds no key,
    so this is an encoded -- not cryptographically protected -- payload: treat
    it as opaque to clients, not as a confidentiality boundary.
    """
    packed = base64.urlsafe_b64encode(zlib.compress(text.encode("utf-8")))
    return _REASONING_STATE_PREFIX + packed.decode("ascii")


def decode_reasoning_state(blob: Any) -> Optional[str]:
    if not isinstance(blob, str) or not blob.startswith(_REASONING_STATE_PREFIX):
        return None
    try:
        raw = base64.urlsafe_b64decode(blob[len(_REASONING_STATE_PREFIX) :])
        return zlib.decompress(raw).decode("utf-8")
    except (ValueError, zlib.error, UnicodeDecodeError):
        return None
