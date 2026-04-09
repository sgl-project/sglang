"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Union

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

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


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
            source: Optional[dict],
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
            request_data["stream_options"] = StreamOptions(include_usage=True)

        chat_request = ChatCompletionRequest(**request_data)

        # Convert tools
        if anthropic_request.tools:
            tools = []
            for tool in anthropic_request.tools:
                tools.append(
                    Tool(
                        type="function",
                        function={
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    )
                )
            chat_request.tools = tools

        # Convert tool choice
        if anthropic_request.tool_choice is not None:
            if anthropic_request.tool_choice.type == "none":
                chat_request.tool_choice = "none"
            elif anthropic_request.tool_choice.type == "auto":
                chat_request.tool_choice = "auto"
            elif anthropic_request.tool_choice.type == "any":
                chat_request.tool_choice = "required"
            elif anthropic_request.tool_choice.type == "tool":
                chat_request.tool_choice = ToolChoice(
                    type="function",
                    function=ToolChoiceFuncName(
                        name=anthropic_request.tool_choice.name
                    ),
                )
        elif anthropic_request.tools:
            # Default to auto when tools are provided
            chat_request.tool_choice = "auto"

        return chat_request

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        received_time = time.time()
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
            return self._error_response(
                status_code=500,
                error_type="internal_error",
                message="Internal processing error",
            )

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
        received_time = time.time()
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
        first_chunk = True
        content_block_index = 0
        content_block_open = False
        finish_reason: Optional[str] = None
        usage_info: Optional[dict] = None
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model

        async for sse_line in openai_stream:
            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                # Close any open content block
                if content_block_open:
                    stop_event = AnthropicStreamEvent(
                        type="content_block_stop",
                        index=content_block_index,
                    )
                    yield _wrap_sse_event(
                        stop_event.model_dump_json(exclude_none=True),
                        "content_block_stop",
                    )

                # Emit message_delta with stop_reason and usage
                stop_reason = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
                delta_event = AnthropicStreamEvent(
                    type="message_delta",
                    delta=AnthropicDelta(stop_reason=stop_reason),
                    usage=AnthropicUsage(
                        input_tokens=(
                            usage_info.get("input_tokens", 0) if usage_info else 0
                        ),
                        output_tokens=(
                            usage_info.get("output_tokens", 0) if usage_info else 0
                        ),
                    ),
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

            # First chunk: emit message_start
            if first_chunk:
                first_chunk = False

                start_event = AnthropicStreamEvent(
                    type="message_start",
                    message=AnthropicMessagesResponse(
                        id=message_id,
                        content=[],
                        model=model,
                        usage=AnthropicUsage(
                            input_tokens=(
                                chunk.usage.prompt_tokens if chunk.usage else 0
                            ),
                            output_tokens=0,
                        ),
                    ),
                )
                yield _wrap_sse_event(
                    start_event.model_dump_json(exclude_none=True),
                    "message_start",
                )
                # Skip if this was just the role chunk with empty content
                if chunk.choices and chunk.choices[0].delta.content == "":
                    continue

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                usage_info = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish reason
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                continue

            delta = choice.delta

            # Handle tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    tc_func = tc.function

                    # New tool call: close previous block, start new one
                    if tc_func and tc_func.name:
                        # Close previous content block if open
                        if content_block_open:
                            stop_event = AnthropicStreamEvent(
                                type="content_block_stop",
                                index=content_block_index,
                            )
                            yield _wrap_sse_event(
                                stop_event.model_dump_json(exclude_none=True),
                                "content_block_stop",
                            )
                            content_block_index += 1

                        # Start tool_use content block
                        start_event = AnthropicStreamEvent(
                            type="content_block_start",
                            index=content_block_index,
                            content_block=AnthropicContentBlock(
                                type="tool_use",
                                id=tc_id or f"toolu_{uuid.uuid4().hex}",
                                name=tc_func.name,
                                input={},
                            ),
                        )
                        yield _wrap_sse_event(
                            start_event.model_dump_json(exclude_none=True),
                            "content_block_start",
                        )
                        content_block_open = True

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
                continue

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                # Start a text content block if needed
                if not content_block_open:
                    start_event = AnthropicStreamEvent(
                        type="content_block_start",
                        index=content_block_index,
                        content_block=AnthropicContentBlock(type="text", text=""),
                    )
                    yield _wrap_sse_event(
                        start_event.model_dump_json(exclude_none=True),
                        "content_block_start",
                    )
                    content_block_open = True

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
        """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response."""
        if not response.choices:
            return AnthropicMessagesResponse(
                content=[AnthropicContentBlock(type="text", text="")],
                model=response.model,
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )

        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

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
            usage=AnthropicUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            ),
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
