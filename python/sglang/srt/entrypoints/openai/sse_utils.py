"""SSE chunk building utilities for OpenAI chat completions streaming."""

from __future__ import annotations

from typing import List, Optional, Union

import msgspec

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class StreamDelta(msgspec.Struct, omit_defaults=True):
    """Delta content for streaming responses.

    Null fields are never emitted (omit_defaults): the Kimi stream spec requires
    reasoning_content to be a string whenever present (P0.12), the end frame to
    be an empty object (P1.10), and increment frames not to mix in unrelated
    null keys (P0.10) — matching OpenAI's own streaming output. Key absence is
    the wire contract's state signal; always serializing null for one SDK's
    attribute convenience fails strict conformance clients.
    """

    reasoning_content: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    # Moonshot extension (P0.5): {"token_ids": [int, ...]}; only carried when
    # the request sets stream_options.include_internal_content=true.
    internal_content: Optional[dict] = None


class StreamChoice(msgspec.Struct):
    """A single choice in a streaming response.

    Must NOT use omit_defaults: the finish_reason key has to appear on every
    frame (null or string, P0.8). usage is only filled on the end frame (P0.4,
    Moonshot extension); middle frames serialize it as null, which P1.15 allows.
    """

    index: int
    delta: StreamDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None
    usage: Optional[dict] = None


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
    internal_content: Optional[dict] = None,
    choice_usage: Optional[dict] = None,
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
    delta = StreamDelta(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
        internal_content=internal_content,
    )
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
        usage=choice_usage,
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
