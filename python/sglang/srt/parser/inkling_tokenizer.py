from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

END_OF_TEXT = "<|endoftext|>"
MESSAGE_USER = "<|message_user|>"
MESSAGE_MODEL = "<|message_model|>"
MESSAGE_SYSTEM = "<|message_system|>"
MESSAGE_TOOL = "<|message_tool|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_IMAGE = "<|content_image|>"
CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_AUDIO_INPUT = "<|content_audio_input|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
CONTENT_XML = "<|content_xml|>"
CONTENT_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
END_MESSAGE = "<|end_message|>"
AUDIO_END = "<|audio_end|>"

IMAGE_TOKEN_ID = -101
AUDIO_TOKEN_ID = -102

INKLING_SPECIAL_TOKEN_IDS: dict[str, int] = {
    END_OF_TEXT: 199999,
    MESSAGE_USER: 200000,
    MESSAGE_MODEL: 200001,
    MESSAGE_SYSTEM: 200002,
    MESSAGE_TOOL: 200003,
    CONTENT_TEXT: 200004,
    CONTENT_IMAGE: 200005,
    CONTENT_MODEL_END_SAMPLING: 200006,
    CONTENT_THINKING: 200008,
    END_MESSAGE: 200010,
    CONTENT_AUDIO_INPUT: 200020,
    CONTENT_TOOL_ERROR: 200022,
    CONTENT_XML: 200024,
    AUDIO_END: 200043,
    CONTENT_INVOKE_TOOL_JSON: 200049,
    CONTENT_INVOKE_TOOL_TEXT: 200057,
}

INKLING_SPECIAL_TOKENS: frozenset[str] = frozenset(INKLING_SPECIAL_TOKEN_IDS)

# The full control alphabet the streaming parsers key on: every framing token
# plus control tokens the model can emit that have no framing-ID mapping.
# The reasoning parser and the tool-call detector MUST share this alphabet —
# a token visible to one but not the other lets malformed headers slip through.
INKLING_CONTROL_TOKENS: frozenset[str] = frozenset(
    {
        *INKLING_SPECIAL_TOKENS,
        "<|content_invoke_tool|>",
        "<|model_trigger_generation|>",
    }
)

INKLING_SPECIAL_TOKEN_NAMES: dict[str, str] = {
    token.removeprefix("<|").removesuffix("|>"): token
    for token in INKLING_SPECIAL_TOKENS
}

ROLE_MESSAGE_TOKENS: dict[str, str] = {
    "user": MESSAGE_USER,
    "assistant": MESSAGE_MODEL,
    "system": MESSAGE_SYSTEM,
    "tool": MESSAGE_TOOL,
}


def normalize_special_token(token: str) -> str:
    """Accept either message_user or <|message_user|> spellings."""
    if token in INKLING_SPECIAL_TOKENS:
        return token
    try:
        return INKLING_SPECIAL_TOKEN_NAMES[token]
    except KeyError as exc:
        raise KeyError(f"unknown Inkling special token: {token!r}") from exc


@dataclass(frozen=True)
class InklingTokenizer:
    """Small wrapper around a base text tokenizer plus Inkling framing IDs.

    Plain text is encoded by the base tokenizer, while the minimal chat
    framing tokens are inserted from the fixed overlay map.
    """

    tokenizer: Any
    special_token_ids: Mapping[str, int] | None = None

    def encode_text(self, text: str) -> list[int]:
        if not isinstance(text, str):
            # Non-string content (e.g. structured content parts, None, or numbers
            # slipping through rendering) should not raise and fail the request;
            # coerce defensively instead.
            text = "" if text is None else str(text)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_special(self, token: str) -> int:
        special = normalize_special_token(token)
        token_ids = self.special_token_ids or INKLING_SPECIAL_TOKEN_IDS
        return int(token_ids[special])

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
