from __future__ import annotations

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
