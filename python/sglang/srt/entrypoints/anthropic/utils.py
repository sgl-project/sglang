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
"""Standalone Anthropic ↔ OpenAI conversion utilities.

External OpenAI-compatible frontends (e.g. token-in-token-out session
routers) must interpret Anthropic Messages requests exactly as this server
does, but without a live serving runtime (tokenizer manager, engine,
streaming loop). This module exposes ``serving.py``'s conversion semantics
as unbound functions while leaving ``serving.py`` and ``protocol.py``
untouched: request and response conversion delegate to the original
``AnthropicServing`` methods through a runtime-detached instance, so there
is one conversion implementation for the server and external callers.

What is new here rather than delegated:

- ``AnthropicRequestContext`` feature gates. ``serving.py`` accepts every
  typed feature and logs/skips what the backend cannot honor; a frontend
  that has only verified a subset end-to-end can instead fail closed with
  an ``AnthropicRequestError`` (HTTP 400) before conversion. Thinking
  (the request param and history blocks) always fails closed: its
  conversion needs the serving runtime's reasoning parser and cannot be
  reproduced data-only.
- ``to_anthropic_error`` returns the Anthropic error envelope as a DTO
  instead of a ``JSONResponse``, and maps 413/422 the way the HTTP layer's
  ``/v1/messages`` exception handler does (``serving.py``'s own
  ``ERROR_TYPE_MAP`` never sees those statuses because FastAPI raises
  them before the handler runs).
- ``to_anthropic_fake_sse_events`` eagerly synthesizes the Anthropic SSE
  event sequence from one complete ``ChatCompletionResponse``. The server
  streams real deltas from the engine; a frontend that proxies a finished
  non-streaming response can still honor a client's ``stream: true`` with
  this one-delta-per-block materialization.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from pydantic import ValidationError

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
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
    is_server_tool,
)
from sglang.srt.entrypoints.anthropic.serving import (
    ERROR_TYPE_MAP as _SERVING_ERROR_TYPE_MAP,
)
from sglang.srt.entrypoints.anthropic.serving import (
    STOP_REASON_MAP,
    AnthropicServing,
    _anthropic_usage_from_openai,
    _scrub_error_message,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

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


class AnthropicRequestError(ValueError):
    """Anthropic request cannot be parsed, validated, or converted (HTTP 400)."""


@dataclass(frozen=True)
class AnthropicRequestContext:
    """Immutable conversion policy for one deployment.

    ``merge_inline_system`` mirrors what ``AnthropicServing.__init__``
    probes from the chat template (``not detect_inline_system_support(...)``).
    The ``allow_*`` gates reject known-but-disabled typed features with
    ``AnthropicRequestError`` before conversion; when a gate is enabled,
    conversion follows ``serving.py`` unchanged. Validation walks only the
    typed request/content/tool models; ``tool_use.input``, custom-tool
    ``input_schema`` and ``metadata`` remain arbitrary-JSON boundaries.
    Thinking has no gate and is always rejected.
    """

    merge_inline_system: bool
    allow_images: bool = False
    allow_output_config: bool = False
    allow_beta_fields: bool = False
    allow_tool_references: bool = False
    allow_search_results: bool = False
    allow_server_tools: bool = False


class _ConversionOnlyAnthropicServing(AnthropicServing):
    """``AnthropicServing`` detached from the serving runtime.

    The conversion methods only read ``self._merge_inline_system``; the
    thinking/reasoning paths additionally need ``self.openai_serving_chat``
    but the feature gates reject those requests before delegation, so
    ``AnthropicServing.__init__`` (which requires a live
    ``OpenAIServingChat``) is deliberately not called.
    """

    def __init__(self, merge_inline_system: bool):
        self.openai_serving_chat = None
        self._merge_inline_system = merge_inline_system


def anthropic_message_id() -> str:
    """Message-ID factory with the same format the server generates."""
    return f"msg_{uuid.uuid4().hex}"


def parse_anthropic_request(body: bytes) -> AnthropicMessagesRequest:
    """Parse the raw request body; failures map to HTTP 400."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise AnthropicRequestError(f"invalid JSON body: {e}") from e
    try:
        return AnthropicMessagesRequest.model_validate(payload)
    except ValidationError as e:
        raise AnthropicRequestError(str(e)) from e


def _iter_typed_content_blocks(request: AnthropicMessagesRequest) -> Iterator[Any]:
    """Yield every typed content block reachable from the request: top-level
    ``system`` blocks, message blocks, and one level of ``tool_result`` nested
    content (the depth ``serving.py``'s conversion reads)."""
    if request.system is not None and not isinstance(request.system, str):
        yield from request.system
    for msg in request.messages:
        if isinstance(msg.content, str):
            continue
        for block in msg.content:
            yield block
            if getattr(block, "type", None) == "tool_result" and isinstance(
                block.content, list
            ):
                yield from block.content


def _validate_known_features(
    request: AnthropicMessagesRequest, context: AnthropicRequestContext
) -> None:
    """Reject known-but-disabled typed features (fail closed, HTTP 400).

    Only typed models are walked; arbitrary-JSON boundaries are not recursed.
    Unknown extra keys inside known models keep the Pydantic ignore behavior
    and are not checked here.
    """
    if request.thinking is not None:
        raise AnthropicRequestError("thinking is not supported by this endpoint")
    if request.output_config is not None and not context.allow_output_config:
        raise AnthropicRequestError("output_config is not enabled for this deployment")
    if request.betas and not context.allow_beta_fields:
        raise AnthropicRequestError("betas is not enabled for this deployment")
    if request.tools and not context.allow_server_tools:
        for tool in request.tools:
            if is_server_tool(tool):
                raise AnthropicRequestError(
                    f"server tool {tool.name!r} (type={tool.type!r}) is not "
                    f"enabled for this deployment"
                )
    for block in _iter_typed_content_blocks(request):
        block_type = getattr(block, "type", None)
        if block_type in ("thinking", "redacted_thinking"):
            raise AnthropicRequestError(
                "thinking content blocks are not supported by this endpoint"
            )
        if block_type == "image" and not context.allow_images:
            raise AnthropicRequestError(
                "image content blocks are not enabled for this deployment"
            )
        if block_type == "tool_reference" and not context.allow_tool_references:
            raise AnthropicRequestError(
                "tool_reference content blocks are not enabled for this deployment"
            )
        if block_type == "search_result" and not context.allow_search_results:
            raise AnthropicRequestError(
                "search_result content blocks are not enabled for this deployment"
            )


def to_openai_request(
    request: AnthropicMessagesRequest, *, context: AnthropicRequestContext
) -> ChatCompletionRequest:
    """Convert an Anthropic Messages request to an OpenAI ChatCompletion
    request under ``context``. Any validation or conversion failure raises
    ``AnthropicRequestError`` (the server's behavior: HTTP 400)."""
    _validate_known_features(request, context)
    converter = _ConversionOnlyAnthropicServing(context.merge_inline_system)
    try:
        return converter._convert_to_chat_completion_request(request)
    except Exception as e:
        # Same policy as ``handle_messages``: every conversion failure is a
        # 400 invalid_request_error, logged with its traceback server-side.
        logger.exception("Error converting Anthropic request: %s", e)
        raise AnthropicRequestError(str(e)) from e


def to_anthropic_response(
    response: ChatCompletionResponse,
    *,
    id_factory: Callable[[], str] = anthropic_message_id,
) -> AnthropicMessagesResponse:
    """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages
    response. ``id_factory`` replaces the delegate's internally generated
    message ID (same wire format by default; inject for determinism)."""
    # ``_convert_response`` never reads the inline-system policy.
    anthropic_response = _ConversionOnlyAnthropicServing(
        merge_inline_system=False
    )._convert_response(response)
    return anthropic_response.model_copy(update={"id": id_factory()})


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
    id_factory: Callable[[], str] = anthropic_message_id,
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
