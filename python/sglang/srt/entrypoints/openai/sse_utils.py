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


def _build_token_id_sse_payload(
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    role: Optional[str],
    content: Optional[str],
    reasoning_content: Optional[str],
    finish_reason: Optional[str],
    logprobs: Optional[dict],
    matched_stop: Union[None, int, str],
    usage: Optional[dict],
    token_ids: Optional[List[int]],
    prompt_token_ids: Optional[List[int]],
) -> dict:
    delta = {"reasoning_content": reasoning_content}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content

    choice = {
        "index": index,
        "delta": delta,
        "logprobs": logprobs,
        "finish_reason": finish_reason,
        "matched_stop": matched_stop,
    }
    if token_ids is not None:
        choice["token_ids"] = token_ids
    if prompt_token_ids is not None:
        choice["prompt_token_ids"] = prompt_token_ids

    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


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
    token_ids: Optional[List[int]] = None,
    prompt_token_ids: Optional[List[int]] = None,
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
        token_ids: Output token ID delta for this choice
        prompt_token_ids: Prompt token IDs, usually only on the first chunk

    Returns:
        SSE-formatted string "data: {...}\\n\\n"
    """
    if token_ids is not None or prompt_token_ids is not None:
        payload = _build_token_id_sse_payload(
            chunk_id=chunk_id,
            created=created,
            model=model,
            index=index,
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            logprobs=logprobs,
            matched_stop=matched_stop,
            usage=usage,
            token_ids=token_ids,
            prompt_token_ids=prompt_token_ids,
        )
        return (_SSE_DATA_B + _stream_encoder.encode(payload) + _SSE_NL_B).decode()

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
