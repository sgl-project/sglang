"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.observability.req_time_stats import monotonic_time

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "request_timeout_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}


def _cached_prompt_tokens(usage) -> int:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _anthropic_input_tokens(usage) -> int:
    return max(
        (getattr(usage, "prompt_tokens", 0) or 0) - _cached_prompt_tokens(usage),
        0,
    )


def _anthropic_usage_from_openai(
    usage,
    *,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool = False,
) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, int] = {}
    cached_tokens = _cached_prompt_tokens(usage)
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        if cached_tokens:
            usage_fields["cache_read_input_tokens"] = cached_tokens
    if include_output:
        usage_fields["output_tokens"] = (
            0 if force_zero_output else (getattr(usage, "completion_tokens", 0) or 0)
        )
    return AnthropicUsage(**usage_fields)


def _wrap_sse_event(data: str, event_type: str) -> str:
    """Format an Anthropic SSE event with event type and data lines."""
    return f"event: {event_type}\ndata: {data}\n\n"


class AnthropicServing:
    """Handler for Anthropic Messages API requests.

    Acts as a translation layer between Anthropic's Messages API and SGLang's
    OpenAI-compatible chat completion infrastructure.
    """

    def __init__(self, openai_serving_chat: OpenAIServingChat):
        self.openai_serving_chat = openai_serving_chat

    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)

    def _convert_to_chat_completion_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert an Anthropic Messages request to an OpenAI ChatCompletion request."""
        openai_messages = []

        def _convert_anthropic_image_source_to_openai_part(
            source: Optional[dict[str, Any] | str],
        ) -> Optional[dict]:
            if not isinstance(source, dict):
                return None

            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                if not data:
                    return None
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }

            url = source.get("url")
            if url:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                }

            return None

        def _text_from_search_result(item: dict[str, Any]) -> str:
            search_parts = []
            title = item.get("title")
            if title:
                search_parts.append(f"Title: {title}")

            source = item.get("source")
            if isinstance(source, dict):
                source_text = source.get("url") or source.get("text")
                if source_text:
                    search_parts.append(f"Source: {source_text}")
            elif source:
                search_parts.append(f"Source: {source}")

            content = item.get("content")
            content_parts = []
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("text"):
                        content_parts.append(part["text"])
            if content_parts:
                search_parts.append("Content: " + "\n".join(content_parts))

            return "\n".join(search_parts)

        def _convert_tool_result_content(
            content: Optional[str | list[dict]],
        ) -> tuple[str | list[dict], str]:
            if isinstance(content, list):
                tool_content_parts = []
                tool_text_parts = []

                for item in content:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            tool_text_parts.append(text)
                            tool_content_parts.append({"type": "text", "text": text})
                    elif item_type == "image":
                        image_part = _convert_anthropic_image_source_to_openai_part(
                            item.get("source")
                        )
                        if image_part is not None:
                            tool_content_parts.append(image_part)
                    elif item_type == "tool_reference":
                        # Anthropic uses `tool_name`; the SGLang chat template
                        # matches on `name`. Translate at the boundary.
                        ref_name = item.get("tool_name") or item.get("name")
                        if ref_name:
                            tool_content_parts.append(
                                {"type": "tool_reference", "name": ref_name}
                            )
                    elif item_type == "search_result":
                        search_text = _text_from_search_result(item)
                        if search_text:
                            tool_text_parts.append(search_text)
                            tool_content_parts.append(
                                {"type": "text", "text": search_text}
                            )

                tool_text = "\n".join(tool_text_parts)
                if (
                    len(tool_content_parts) == 1
                    and tool_content_parts[0]["type"] == "text"
                ):
                    return tool_content_parts[0]["text"], tool_text
                if tool_content_parts:
                    return tool_content_parts, tool_text
                return "", tool_text

            tool_text = str(content) if content else ""
            return tool_text, tool_text

        # Add system message if provided
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                openai_messages.append(
                    {"role": "system", "content": anthropic_request.system}
                )
            else:
                system_parts = []
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_parts.append(block.text)
                system_text = "\n".join(system_parts)
                openai_messages.append({"role": "system", "content": system_text})

        # Convert messages
        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                openai_messages.append({"role": msg.role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": msg.role}
            content_parts = []
            tool_calls = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    content_parts.append({"type": "text", "text": block.text})

                elif block.type == "image" and block.source:
                    image_part = _convert_anthropic_image_source_to_openai_part(
                        block.source
                    )
                    if image_part is not None:
                        content_parts.append(image_part)

                elif block.type == "search_result":
                    search_text = _text_from_search_result(block.model_dump())
                    if search_text:
                        content_parts.append({"type": "text", "text": search_text})

                elif block.type == "tool_use":
                    tool_call = {
                        "id": block.id or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                    tool_calls.append(tool_call)

                elif block.type == "tool_result":
                    tool_content, tool_text = _convert_tool_result_content(
                        block.content
                    )

                    # Use tool_use_id (per spec) with fallback to id
                    tool_call_id = block.tool_use_id or block.id or ""

                    # Tool results from user become separate tool messages
                    if msg.role == "user":
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }
                        )
                    else:
                        content_parts.append(
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_text}",
                            }
                        )

            # Attach tool calls to assistant messages
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            # Attach content
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_msg["content"] = content_parts[0]["text"]
                else:
                    openai_msg["content"] = content_parts
            elif not tool_calls:
                continue

            openai_messages.append(openai_msg)

        # Build ChatCompletionRequest
        request_data = {
            "messages": openai_messages,
            "model": anthropic_request.model,
            "max_tokens": anthropic_request.max_tokens,
            "stream": anthropic_request.stream or False,
        }

        if anthropic_request.temperature is not None:
            request_data["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            request_data["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            request_data["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences is not None:
            request_data["stop"] = anthropic_request.stop_sequences

        # Enable usage in stream so we can report it
        if anthropic_request.stream:
            request_data["stream_options"] = StreamOptions(
                include_usage=True,
                continuous_usage_stats=True,
            )

        chat_request = ChatCompletionRequest(**request_data)

        # Convert tools. Deferred tools stay in the list with defer_loading=True;
        # the chat template hides them from the initial <tools> block and renders
        # them on demand when a tool_reference block names them.
        if anthropic_request.tools:
            converted_tools = []
            for tool in anthropic_request.tools:
                tool_type = tool.type or ""
                is_builtin_server_tool = tool_type.startswith(
                    "web_search"
                ) or tool.name.startswith("web_search")
                if tool.input_schema is None:
                    if is_builtin_server_tool:
                        continue
                    raise ValueError("input_schema is required for custom tools")

                converted_tools.append(
                    Tool(
                        type="function",
                        defer_loading=tool.defer_loading,
                        function={
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    )
                )

            if converted_tools:
                chat_request.tools = converted_tools

        # Convert tool choice
        if anthropic_request.tool_choice is not None:
            tc_type = anthropic_request.tool_choice.type
            if tc_type == "none":
                chat_request.tool_choice = "none"
            elif chat_request.tools:
                if tc_type == "auto":
                    chat_request.tool_choice = "auto"
                elif tc_type == "any":
                    chat_request.tool_choice = "required"
                elif tc_type == "tool":
                    chat_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(
                            name=anthropic_request.tool_choice.name
                        ),
                    )
        elif chat_request.tools:
            chat_request.tool_choice = "auto"

        return chat_request

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        received_time = monotonic_time()
        received_time_perf = time.perf_counter()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            # Convert to internal request
            validation_time = time.perf_counter() - received_time_perf
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.validation_time = validation_time
            adapted_request.received_time = received_time
            adapted_request.received_time_perf = received_time_perf

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        except Exception as e:
            logger.exception("Error processing Anthropic request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )

        # Check for error responses from OpenAI handler
        if not isinstance(response, ChatCompletionResponse):
            # It's an error response (ORJSONResponse)
            return self._convert_openai_error_response(response)

        # Convert to Anthropic response
        anthropic_response = self._convert_response(response)
        return JSONResponse(content=anthropic_response.model_dump(exclude_none=True))

    async def _handle_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Handle streaming Anthropic request."""
        received_time = monotonic_time()
        received_time_perf = time.perf_counter()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            validation_time = time.perf_counter() - received_time_perf
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.validation_time = validation_time
            adapted_request.received_time = received_time
            adapted_request.received_time_perf = received_time_perf
        except Exception as e:
            logger.exception("Error converting streaming request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )

        return StreamingResponse(
            self._generate_anthropic_stream(
                adapted_request,
                processed_request,
                anthropic_request,
                raw_request,
            ),
            media_type="text/event-stream",
            background=self.openai_serving_chat.tokenizer_manager.create_abort_task(
                adapted_request
            ),
        )

    async def _generate_anthropic_stream(
        self,
        adapted_request,
        processed_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI chat stream to Anthropic event stream."""
        openai_stream = self.openai_serving_chat._generate_chat_stream(
            adapted_request, processed_request, raw_request
        )

        # State tracking
        content_block_index = 0
        content_block_open = False
        content_block_type: Optional[str] = None
        finish_reason: Optional[str] = None
        final_usage: Optional[AnthropicUsage] = None
        message_started = False
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model

        def _message_start_event(usage) -> AnthropicStreamEvent:
            return AnthropicStreamEvent(
                type="message_start",
                message=AnthropicMessagesResponse(
                    id=message_id,
                    content=[],
                    model=model,
                    usage=_anthropic_usage_from_openai(
                        usage,
                        include_input=True,
                        include_output=True,
                        force_zero_output=True,
                    ),
                ),
            )

        def _close_content_block_events() -> list[tuple[str, AnthropicStreamEvent]]:
            nonlocal content_block_index, content_block_open, content_block_type

            events = []
            if not content_block_open:
                return events

            if content_block_type == "thinking":
                events.append(
                    (
                        "content_block_delta",
                        AnthropicStreamEvent(
                            type="content_block_delta",
                            index=content_block_index,
                            delta=AnthropicDelta(
                                type="signature_delta",
                                signature="",
                            ),
                        ),
                    )
                )

            events.append(
                (
                    "content_block_stop",
                    AnthropicStreamEvent(
                        type="content_block_stop",
                        index=content_block_index,
                    ),
                )
            )
            content_block_open = False
            content_block_type = None
            content_block_index += 1
            return events

        def _ensure_content_block_events(
            block_type: str,
            content_block: AnthropicContentBlock,
        ) -> list[tuple[str, AnthropicStreamEvent]]:
            nonlocal content_block_open, content_block_type

            events = []
            if content_block_open and content_block_type != block_type:
                events.extend(_close_content_block_events())
            if not content_block_open:
                events.append(
                    (
                        "content_block_start",
                        AnthropicStreamEvent(
                            type="content_block_start",
                            index=content_block_index,
                            content_block=content_block,
                        ),
                    )
                )
                content_block_open = True
                content_block_type = block_type
            return events

        async for sse_line in openai_stream:
            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                if not message_started:
                    start_event = _message_start_event(None)
                    yield _wrap_sse_event(
                        start_event.model_dump_json(exclude_none=True),
                        "message_start",
                    )
                    message_started = True

                # Close any open content block
                for event_name, event in _close_content_block_events():
                    yield _wrap_sse_event(
                        event.model_dump_json(exclude_none=True),
                        event_name,
                    )

                # Emit message_delta with stop_reason and usage
                stop_reason = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
                delta_event = AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=stop_reason),
                    usage=final_usage or AnthropicUsage(output_tokens=0),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "message_delta",
                )

                # Emit message_stop
                stop_msg = AnthropicStreamEvent(type="message_stop")
                yield _wrap_sse_event(
                    stop_msg.model_dump_json(exclude_none=True),
                    "message_stop",
                )
                continue

            # Parse the OpenAI chunk
            try:
                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
            except Exception:
                logger.debug("Failed to parse stream chunk: %s", data_str)
                error_event = AnthropicStreamEvent(
                    type="error",
                    error=AnthropicError(
                        type="api_error", message="Stream processing error"
                    ),
                )
                yield _wrap_sse_event(
                    error_event.model_dump_json(exclude_none=True), "error"
                )
                continue

            if chunk.usage is not None:
                final_usage = _anthropic_usage_from_openai(
                    chunk.usage,
                    include_input=False,
                    include_output=True,
                )

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish reason
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                if not message_started:
                    start_event = _message_start_event(chunk.usage)
                    yield _wrap_sse_event(
                        start_event.model_dump_json(exclude_none=True),
                        "message_start",
                    )
                    message_started = True
                continue

            delta = choice.delta

            has_delta_payload = bool(
                delta.reasoning_content
                or delta.tool_calls
                or (delta.content is not None and delta.content != "")
                or chunk.usage
            )
            if has_delta_payload and not message_started:
                start_event = _message_start_event(chunk.usage)
                yield _wrap_sse_event(
                    start_event.model_dump_json(exclude_none=True),
                    "message_start",
                )
                message_started = True

            if (
                not has_delta_payload
                and delta.role == "assistant"
                and (delta.content is None or delta.content == "")
            ):
                continue

            # Handle reasoning content deltas
            if delta.reasoning_content:
                for event_name, event in _ensure_content_block_events(
                    "thinking",
                    AnthropicContentBlock(
                        type="thinking",
                        thinking="",
                        signature="",
                    ),
                ):
                    yield _wrap_sse_event(
                        event.model_dump_json(exclude_none=True),
                        event_name,
                    )

                delta_event = AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta=AnthropicDelta(
                        type="thinking_delta",
                        thinking=delta.reasoning_content,
                    ),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "content_block_delta",
                )

            # Handle tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    tc_func = tc.function

                    # New tool call: close previous block, start new one
                    if tc_func and tc_func.name:
                        # Start tool_use content block
                        for event_name, event in _ensure_content_block_events(
                            "tool_use",
                            AnthropicContentBlock(
                                type="tool_use",
                                id=tc_id or f"toolu_{uuid.uuid4().hex}",
                                name=tc_func.name,
                                input={},
                            ),
                        ):
                            yield _wrap_sse_event(
                                event.model_dump_json(exclude_none=True),
                                event_name,
                            )

                        # Stream initial arguments if present
                        if tc_func.arguments:
                            delta_event = AnthropicStreamEvent(
                                type="content_block_delta",
                                index=content_block_index,
                                delta=AnthropicDelta(
                                    type="input_json_delta",
                                    partial_json=tc_func.arguments,
                                ),
                            )
                            yield _wrap_sse_event(
                                delta_event.model_dump_json(exclude_none=True),
                                "content_block_delta",
                            )

                    elif tc_func and tc_func.arguments:
                        # Continuing arguments for current tool call
                        if content_block_type != "tool_use":
                            continue
                        delta_event = AnthropicStreamEvent(
                            type="content_block_delta",
                            index=content_block_index,
                            delta=AnthropicDelta(
                                type="input_json_delta",
                                partial_json=tc_func.arguments,
                            ),
                        )
                        yield _wrap_sse_event(
                            delta_event.model_dump_json(exclude_none=True),
                            "content_block_delta",
                        )

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                # Start a text content block if needed
                for event_name, event in _ensure_content_block_events(
                    "text",
                    AnthropicContentBlock(type="text", text=""),
                ):
                    yield _wrap_sse_event(
                        event.model_dump_json(exclude_none=True),
                        event_name,
                    )

                # Emit text delta
                delta_event = AnthropicStreamEvent(
                    type="content_block_delta",
                    index=content_block_index,
                    delta=AnthropicDelta(
                        type="text_delta",
                        text=delta.content,
                    ),
                )
                yield _wrap_sse_event(
                    delta_event.model_dump_json(exclude_none=True),
                    "content_block_delta",
                )

    def _convert_response(
        self, response: ChatCompletionResponse
    ) -> AnthropicMessagesResponse:
        """Convert an OpenAI ChatCompletionResponse to an Anthropic response."""
        if not response.choices:
            return AnthropicMessagesResponse(
                content=[AnthropicContentBlock(type="text", text="")],
                model=response.model,
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )

        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

        # Add reasoning content
        if choice.message.reasoning_content:
            content.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=choice.message.reasoning_content,
                    signature="",
                )
            )

        # Add text content
        if choice.message.content:
            content.append(
                AnthropicContentBlock(type="text", text=choice.message.content)
            )

        # Add tool calls
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}

                content.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        # Map stop reason
        stop_reason = STOP_REASON_MAP.get(choice.finish_reason or "stop", "end_turn")

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex}",
            content=content,
            model=response.model,
            stop_reason=stop_reason,
            usage=_anthropic_usage_from_openai(
                response.usage,
                include_input=True,
                include_output=True,
            ),
        )

    def _convert_openai_error_response(self, response) -> JSONResponse:
        status_code = getattr(response, "status_code", 500)
        error_type = ERROR_TYPE_MAP.get(status_code, "api_error")
        message = "Internal processing error"

        body = getattr(response, "body", None)
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = None

            if isinstance(payload, dict):
                error_payload = payload.get("error", payload)
                if isinstance(error_payload, dict):
                    message = (
                        error_payload.get("message")
                        or payload.get("message")
                        or message
                    )
                    error_type = error_payload.get("type") or error_type
                elif isinstance(error_payload, str):
                    message = error_payload
                elif payload.get("message"):
                    message = payload["message"]

        return self._error_response(
            status_code=status_code,
            error_type=error_type,
            message=message,
        )

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
    ) -> JSONResponse:
        """Create an Anthropic-format error response."""
        error_resp = AnthropicErrorResponse(
            error=AnthropicError(type=error_type, message=message)
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    async def handle_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle /v1/messages/count_tokens endpoint.

        Converts the request to a ChatCompletionRequest, applies the chat
        template via the OpenAI handler to tokenize, and returns the count.
        """
        try:
            # Build a minimal AnthropicMessagesRequest so we can reuse conversion
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,  # dummy, not used for counting
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            chat_request = self._convert_to_chat_completion_request(messages_request)
        except Exception as e:
            logger.exception("Error converting count_tokens request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        try:
            is_multimodal = (
                self.openai_serving_chat.tokenizer_manager.model_config.is_multimodal
            )
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
        except Exception as e:
            logger.exception("Error counting tokens: %s", e)
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal server error",
            )
