"""SSE chunk building utilities for OpenAI chat completions streaming."""

from __future__ import annotations

from typing import List, Optional, Union

import msgspec

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class StreamDelta(msgspec.Struct, omit_defaults=True):
    """Delta content for streaming responses.

    OpenAI Python SDK's ChoiceDelta does not declare reasoning_content; it is
    surfaced via pydantic `extra`. With omit_defaults=True, defaulting to
    None would drop the key entirely from the SSE payload, making
    `data.reasoning_content` raise AttributeError on the client. Keep it
    required (no default) so it is always serialized as null or a string.
    """

    reasoning_content: Optional[str]
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(msgspec.Struct):
    """A single choice in a streaming response."""

    index: int
    delta: StreamDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None


class StreamChunk(msgspec.Struct, omit_defaults=True):
    """A complete streaming chunk."""

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[dict] = None


_stream_encoder = msgspec.json.Encoder()


def build_sse_content(
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    role: Optional[str] = None,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[dict] = None,
    matched_stop: Union[None, int, str] = None,
    usage: Optional[dict] = None,
) -> str:
    """Build an SSE chunk string for content/reasoning updates.

    Args:
        chunk_id: Request ID for this chunk
        created: Unix timestamp
        model: Model name
        index: Choice index
        role: Message role (usually "assistant")
        content: Text content delta
        reasoning_content: Reasoning/thinking content delta
        finish_reason: Finish reason if done
        logprobs: Log probabilities if requested
        matched_stop: Stop token/string that was matched
        usage: Token usage statistics

    Returns:
        SSE-formatted string "data: {...}\\n\\n"
    """
    delta = StreamDelta(role=role, content=content, reasoning_content=reasoning_content)
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
    )
    chunk = StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()
