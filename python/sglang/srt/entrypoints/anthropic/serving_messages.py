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
"""Anthropic Messages API serving handler"""

import copy
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import uuid

import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicErrorResponse,
    AnthropicError,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicTool,
    AnthropicUsage,
)
from sglang.srt.entrypoints.openai.protocol import MessageProcessingResult
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.jinja_template_utils import process_content_for_template_format
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class AnthropicServingMessages(OpenAIServingBase):
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self, tokenizer_manager: TokenizerManager, template_manager: TemplateManager
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "msg_"

    def _convert_anthropic_to_openai_messages(
        self,
        anthropic_messages: List[AnthropicMessage],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic message format to OpenAI format"""
        openai_messages = []

        # Add system message if provided
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in anthropic_messages:
            openai_msg = {"role": msg.role}

            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # Handle complex content blocks
                content_parts = []
                tool_calls = []

                for block in msg.content:
                    if block.type == "text" and block.text:
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.source:
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": block.source.get("data", "")},
                            }
                        )
                    elif block.type == "tool_use":
                        # Convert tool use to function call format
                        tool_call = {
                            "id": block.id or f"call_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input or {}),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif block.type == "tool_result":
                        # For tool results, we need to create a tool message
                        # This will be handled separately as a tool response message
                        if msg.role == "user":
                            # Tool result from user should be converted to tool message
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.id,
                                    "content": str(block.content)
                                    if block.content
                                    else "",
                                }
                            )
                        else:
                            # Assistant tool result becomes regular text
                            content_parts.append(
                                {
                                    "type": "text",
                                    "text": f"Tool result: {str(block.content) if block.content else ''}",
                                }
                            )

                # Add tool calls to the message if any
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                # Add content parts if any
                if content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_msg["content"] = content_parts[0]["text"]
                    else:
                        openai_msg["content"] = content_parts
                elif not tool_calls:
                    # If no content and no tool calls, add empty content
                    openai_msg["content"] = ""

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_to_internal_request(
        self, request: AnthropicMessagesRequest
    ) -> GenerateReqInput:
        """Convert Anthropic request to SGLang internal format"""

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)

        # Build sampling parameters
        sampling_params = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
        }

        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k

        if request.stop_sequences:
            sampling_params["stop"] = request.stop_sequences

        # Handle single vs multiple requests
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        return GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            stream=request.stream or False,
            return_logprob=False,
            rid=f"anthropic_{int(time.time() * 1000)}",
        )

    def _process_messages(
        self, request: AnthropicMessagesRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice:
            tools = [item.model_dump() for item in request.tools]

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        return result

    def _apply_jinja_template(
        self,
        request: AnthropicMessagesRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        for message in request.messages:
            if message.content is None:
                message.content = ""
            msg_dict = message.model_dump()

            # Process content based on detected template format
            processed_msg = process_content_for_template_format(
                msg_dict,
                template_content_format,
                image_data,
                video_data,
                audio_data,
                modalities,
            )
            openai_compatible_messages.append(processed_msg)

        # Handle assistant prefix for continue_final_message
        assistant_prefix = None
        if (
            openai_compatible_messages
            and openai_compatible_messages[-1]["role"] == "assistant"
        ):
            assistant_prefix = openai_compatible_messages[-1]["content"]
            openai_compatible_messages = openai_compatible_messages[:-1]

        try:
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
            )
        except Exception:
            #  This except branch will be triggered when the chosen model
            #  has a different tools input format that is not compatible
            #  with openAI's apply_chat_template tool_call format, like Mistral.
            tools = (
                [t if "function" in t else {"function": t} for t in tools]
                if tools
                else None
            )
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
            )

        if assistant_prefix:
            encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
            if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
                encoded = encoded[1:]
            prompt_ids += encoded

        if is_multimodal:
            prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop_sequences
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: AnthropicMessagesRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if request.messages and request.messages[-1].role == "assistant":
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = []

        if request.stop_sequences:
            if isinstance(request.stop_sequences, str):
                stop.append(request.stop_sequences)
            else:
                stop.extend(request.stop_sequences)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _create_anthropic_content_block(self, text: str) -> AnthropicContentBlock:
        """Create Anthropic content block from text"""
        return AnthropicContentBlock(type="text", text=text)

    def _create_anthropic_response(
        self,
        request_id: str,
        model: str,
        content: str,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str = "end_turn",
    ) -> AnthropicMessagesResponse:
        """Create Anthropic Messages response"""
        usage = AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        content_blocks = [self._create_anthropic_content_block(content)]

        return AnthropicMessagesResponse(
            id=request_id,
            content=content_blocks,
            model=model,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _convert_anthropic_tools_to_openai(
        self,
        tools: List[AnthropicTool],
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert Anthropic tools to OpenAI function format"""
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            openai_tools.append(openai_tool)

        result = {"tools": openai_tools}

        # Handle tool_choice conversion
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    result["tool_choice"] = "auto"
                elif tool_choice == "required":
                    result["tool_choice"] = "required"
                elif tool_choice == "none":
                    result["tool_choice"] = "none"
            elif isinstance(tool_choice, dict) and "name" in tool_choice:
                # Specific tool choice
                result["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }

        return result

    def _convert_function_calls_to_anthropic(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[AnthropicContentBlock]:
        """Convert OpenAI function calls to Anthropic tool_use blocks"""
        content_blocks = []

        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_use_block = AnthropicContentBlock(
                    type="tool_use",
                    id=tool_call.get("id"),
                    name=function.get("name"),
                    input=arguments,
                )
                content_blocks.append(tool_use_block)

        return content_blocks

    def _create_stream_event(self, event_type: str, **kwargs) -> AnthropicStreamEvent:
        """Create streaming event"""
        return AnthropicStreamEvent(type=event_type, **kwargs)

    async def _generate_stream_response(
        self,
        internal_request: GenerateReqInput,
        anthropic_request: AnthropicMessagesRequest,
        request_id: str,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response in Anthropic format"""

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}

        # Send message_start event
        start_message = AnthropicMessagesResponse(
            id=request_id,
            content=[],
            model=anthropic_request.model,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )

        start_event = self._create_stream_event("message_start", message=start_message)
        yield f"event: message_start\ndata: {orjson.dumps(start_event.model_dump()).decode()}\n\n"

        # Send content_block_start event
        content_start_event = self._create_stream_event(
            "content_block_start",
            index=0,
            content_block=AnthropicContentBlock(type="text", text=""),
        )
        yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event.model_dump()).decode()}\n\n"

        accumulated_text = ""
        input_tokens = 0
        output_tokens = 0

        try:
            async for content in self.tokenizer_manager.generate_request(
                internal_request, raw_request
            ):
                index = content.get("index", 0)
                if "text" in content:
                    # Process content delta
                    stream_buffer = stream_buffers.get(index, "")
                    text_delta = content["text"][len(stream_buffer) :]
                    stream_buffers[index] = stream_buffer + text_delta

                    accumulated_text += text_delta

                    # Send content_block_delta event
                    delta = AnthropicDelta(type="text_delta", text=text_delta)
                    delta_event = self._create_stream_event(
                        "content_block_delta", index=0, delta=delta
                    )
                    yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event.model_dump()).decode()}\n\n"

                if "meta_info" in content:
                    meta = content["meta_info"]
                    input_tokens = meta.get("prompt_tokens", 0)
                    output_tokens = meta.get("completion_tokens", 0)

        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            error_event = self._create_stream_event(
                "error",
                error=AnthropicError(type="internal_server_error", message=str(e)),
            )
            yield f"event: error\ndata: {orjson.dumps(error_event.model_dump()).decode()}\n\n"
            return

        # Send content_block_stop event
        stop_event = self._create_stream_event("content_block_stop", index=0)
        yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"

        # Send message_stop event with final usage
        final_usage = AnthropicUsage(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        final_message = AnthropicMessagesResponse(
            id=request_id,
            content=[self._create_anthropic_content_block(accumulated_text)],
            model=anthropic_request.model,
            usage=final_usage,
        )

        stop_message_event = self._create_stream_event(
            "message_stop", message=final_message
        )
        yield f"event: message_stop\ndata: {orjson.dumps(stop_message_event.model_dump()).decode()}\n\n"

    async def handle_request(
        self, request: AnthropicMessagesRequest, raw_request: Request
    ) -> Union[AnthropicMessagesResponse, StreamingResponse, ORJSONResponse]:
        """Handle Anthropic Messages API request"""

        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"

        try:
            # Convert to internal format
            internal_request = self._convert_to_internal_request(request)

            if request.stream:
                # Return streaming response
                stream_generator = self._generate_stream_response(
                    internal_request, request, request_id, raw_request
                )
                return StreamingResponse(
                    stream_generator,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Generate single response
                response_generator = self.tokenizer_manager.generate_request(
                    internal_request, raw_request
                )

                full_response = await response_generator.__anext__()

                content = full_response.get("text", "")
                meta_info = full_response.get("meta_info", {})
                input_tokens = meta_info.get("prompt_tokens", 0)
                output_tokens = meta_info.get("completion_tokens", 0)

                # Check if response contains function calls
                content_blocks = []

                # Look for function calls in the response
                if "function_calls" in full_response or "tool_calls" in full_response:
                    tool_calls = full_response.get(
                        "function_calls"
                    ) or full_response.get("tool_calls", [])
                    if tool_calls:
                        # Convert function calls to Anthropic tool_use blocks
                        tool_blocks = self._convert_function_calls_to_anthropic(
                            tool_calls
                        )
                        content_blocks.extend(tool_blocks)

                        # Set stop reason to tool_use
                        stop_reason = "tool_use"
                    else:
                        # Regular text response
                        content_blocks = [self._create_anthropic_content_block(content)]
                        stop_reason = "end_turn"
                else:
                    # Regular text response
                    content_blocks = [self._create_anthropic_content_block(content)]
                    stop_reason = "end_turn"

                # Check for other stop reasons
                if meta_info.get("finish_reason") == "length":
                    stop_reason = "max_tokens"
                elif meta_info.get("finish_reason") == "stop":
                    stop_reason = "stop_sequence"

                response = AnthropicMessagesResponse(
                    id=request_id,
                    content=content_blocks,
                    model=request.model,
                    stop_reason=stop_reason,
                    usage=AnthropicUsage(
                        input_tokens=input_tokens, output_tokens=output_tokens
                    ),
                )

                return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            error_response = AnthropicErrorResponse(
                error=AnthropicError(type="invalid_request_error", message=str(e))
            )
            return ORJSONResponse(status_code=400, content=error_response.model_dump())

        except Exception as e:
            logger.error(f"Internal error: {get_exception_traceback()}")
            error_response = AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_server_error", message="Internal server error"
                )
            )
            return ORJSONResponse(status_code=500, content=error_response.model_dump())
