"""SSE chunk building utilities for OpenAI chat completions streaming."""

from __future__ import annotations

from typing import List, Optional, Union

import msgspec

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class StreamDelta(msgspec.Struct, omit_defaults=True):
    """Delta content for streaming responses.

    None 字段一律不下发(omit_defaults):Kimi 流式规范要求 reasoning_content
    出现即必须是字符串(P0.12)、结束帧 delta 为空对象(P1.10)、增量帧不混入
    无关 null 键(P0.10)—— OpenAI 官方流式输出同样不发 null 字段。此前为让
    OpenAI Python SDK 的 `data.reasoning_content` 属性恒存在而强制序列化
    null;但键缺失本就是流式契约的状态信号,客户端应按键存在与否分支,为单一
    SDK 的取值便利违反 wire 契约会被严格客户端(KVV stream-spec)判协议违规。
    """

    reasoning_content: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    # Moonshot 扩展(P0.5):{"token_ids": [int, ...]},仅请求
    # stream_options.include_internal_content=true 时携带。
    internal_content: Optional[dict] = None


class StreamChoice(msgspec.Struct):
    """A single choice in a streaming response.

    注意不能开 omit_defaults:finish_reason 键必须每帧出现(null 或字符串,
    P0.8)。usage 仅结束帧填充(P0.4,Moonshot 扩展),中间帧序列化为
    null —— P1.15 允许中间帧 usage 为 null 或缺失。
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
