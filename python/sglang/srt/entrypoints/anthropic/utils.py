# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Anthropic error conversion and fake-SSE response helpers.

Request and response format conversion lives in serving.py as public,
runtime-independent functions. This module contains the two adaptations used
by external frontends: an Anthropic error DTO and an eager event sequence for
an already-complete OpenAI response.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Callable, Optional

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessageEndDelta,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
)
from sglang.srt.entrypoints.anthropic.serving import (
    ERROR_TYPE_MAP as _SERVING_ERROR_TYPE_MAP,
)
from sglang.srt.entrypoints.anthropic.serving import (
    STOP_REASON_MAP,
    _anthropic_usage_from_openai,
    _scrub_error_message,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

logger = logging.getLogger(__name__)

# ``serving.py``'s map plus the two statuses the HTTP layer maps in its
# ``/v1/messages`` exception handler (``http_server.py``): FastAPI raises
# 413/422 before ``AnthropicServing`` runs, so ``serving.py`` never needed
# them. Unlisted statuses fall through to ``api_error`` in both sources.
ERROR_TYPE_MAP = {
    **_SERVING_ERROR_TYPE_MAP,
    413: "request_too_large",
    422: "invalid_request_error",
}


def _anthropic_message_id() -> str:
    """Message-ID factory with the same format the server generates."""
    return f"msg_{uuid.uuid4().hex}"


def to_anthropic_error(status_code: int, body: bytes) -> AnthropicErrorResponse:
    """Map an upstream error status and body to an Anthropic error envelope.

    Same payload parsing and scrub policy as ``serving.py``'s
    ``_convert_openai_error_response``, but against the composite
    ``ERROR_TYPE_MAP`` above and returning the envelope DTO — HTTP response
    construction stays with the caller. 4xx keeps a sanitized upstream
    message and honors an upstream ``error.type``; 5xx is always the
    generic scrubbed message.
    """
    body = body or b""
    error_type = ERROR_TYPE_MAP.get(status_code, "api_error")

    upstream_message: Optional[str] = None
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Non-JSON body (HTML gateway error, plain text, ...). Use a bounded
        # slice of the raw body so the client still has a useful hint.
        upstream_message = body.decode("utf-8", errors="replace")[:500]
    else:
        if isinstance(payload, dict):
            error_payload = payload.get("error", payload)
            if isinstance(error_payload, dict):
                upstream_message = error_payload.get("message") or payload.get(
                    "message"
                )
                # Honor the upstream error.type only for 4xx; 5xx is
                # normalized by the scrub below.
                if status_code < 500:
                    upstream_type = error_payload.get("type")
                    if isinstance(upstream_type, str) and upstream_type:
                        error_type = upstream_type
            elif isinstance(error_payload, str):
                upstream_message = error_payload
            elif isinstance(payload.get("message"), str):
                upstream_message = payload["message"]

    message = _scrub_error_message(upstream_message or "", status_code)
    return AnthropicErrorResponse(
        error=AnthropicError(type=error_type, message=message)
    )


def to_anthropic_fake_sse_events(
    response: ChatCompletionResponse,
    *,
    model: str,
    id_factory: Callable[[], str] = _anthropic_message_id,
) -> tuple[AnthropicStreamEvent, ...]:
    """Eagerly synthesize the Anthropic SSE event sequence from one complete
    OpenAI response.

    Event schema, block ordering (thinking → text → tool_use), index
    accounting, usage split (input on ``message_start``, output on
    ``message_delta``) and stop-reason mapping follow the server's live
    stream; collapsing each block into a single delta is the documented
    difference. ``model`` must be the original Anthropic request model, not
    the backend's possibly-aliased response model.
    """
    events: list[AnthropicStreamEvent] = [
        MessageStartEvent(
            message=AnthropicMessagesResponse(
                id=id_factory(),
                content=[],
                model=model,
                usage=_anthropic_usage_from_openai(
                    response.usage,
                    include_input=True,
                    include_output=True,
                    force_zero_output=True,
                ),
            )
        )
    ]

    message = response.choices[0].message if response.choices else None
    index = 0

    if message is not None and message.reasoning_content:
        events.append(
            ContentBlockStartEvent(
                index=index, content_block=ThinkingBlock(thinking="")
            )
        )
        events.append(
            ContentBlockDeltaEvent(
                index=index, delta=ThinkingDelta(thinking=message.reasoning_content)
            )
        )
        events.append(ContentBlockStopEvent(index=index))
        index += 1

    if message is not None and message.content:
        events.append(
            ContentBlockStartEvent(index=index, content_block=TextBlock(text=""))
        )
        events.append(
            ContentBlockDeltaEvent(index=index, delta=TextDelta(text=message.content))
        )
        events.append(ContentBlockStopEvent(index=index))
        index += 1

    if message is not None and message.tool_calls:
        for tool_call in message.tool_calls:
            events.append(
                ContentBlockStartEvent(
                    index=index,
                    content_block=ToolUseBlock(
                        id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                        name=tool_call.function.name,
                        input={},
                    ),
                )
            )
            if tool_call.function.arguments:
                events.append(
                    ContentBlockDeltaEvent(
                        index=index,
                        delta=InputJsonDelta(partial_json=tool_call.function.arguments),
                    )
                )
            events.append(ContentBlockStopEvent(index=index))
            index += 1

    finish_reason = (
        (response.choices[0].finish_reason or "stop") if response.choices else "stop"
    )
    if finish_reason not in STOP_REASON_MAP:
        logger.warning(
            "Unmapped OpenAI finish_reason %r; defaulting to end_turn", finish_reason
        )
    events.append(
        MessageDeltaEvent(
            delta=AnthropicMessageEndDelta(
                stop_reason=STOP_REASON_MAP.get(finish_reason, "end_turn")
            ),
            usage=_anthropic_usage_from_openai(
                response.usage, include_input=False, include_output=True
            ),
        )
    )
    events.append(MessageStopEvent())
    return tuple(events)
