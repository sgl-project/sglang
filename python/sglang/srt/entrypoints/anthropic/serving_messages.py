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

import json
import logging
import time
import uuid
from abc import ABC
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicFinishReason,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicUsage,
    TextContentBlock,
    ThinkingContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    MessageProcessingResult,
    Tool,
    ToolCallConstraint,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class AnthropicServingMessages(ABC):
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self, tokenizer_manager: TokenizerManager, template_manager: TemplateManager
    ):
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser

    async def handle_request(
        self, request: AnthropicMessagesRequest, raw_request: Request
    ) -> Union[AnthropicMessagesResponse, StreamingResponse, ORJSONResponse]:
        """Handle Anthropic Messages API request."""
        try:
            internal_request = self._convert_to_internal_request(request)

            if request.stream:
                return StreamingResponse(
                    self._handle_streaming_request(
                        internal_request, request, raw_request
                    ),
                    media_type="text/event-stream",
                )
            else:
                response = await self._handle_non_streaming_request(
                    internal_request, request, raw_request
                )
                return ORJSONResponse(content=response.model_dump())
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequestError",
                status_code=400,
            )
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    async def handle_count_tokens_request(
        self, request: AnthropicCountTokensRequest, raw_request: Request
    ) -> Union[AnthropicCountTokensResponse, ORJSONResponse]:
        """Handle Anthropic Count Tokens API request.

        Counts tokens without generating a response.
        """
        try:
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )

            is_multimodal = self.tokenizer_manager.model_config.is_multimodal
            processed = self._process_messages(messages_request, is_multimodal)

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            elif isinstance(processed.prompt_ids, str):
                input_tokens = len(
                    self.tokenizer_manager.tokenizer.encode(processed.prompt_ids)
                )
            else:
                input_tokens = 0

            response = AnthropicCountTokensResponse(input_tokens=input_tokens)
            return ORJSONResponse(content=response.model_dump())

        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except ValueError as e:
            return self.create_error_response(
                message=str(e),
                err_type="BadRequestError",
                status_code=400,
            )
        except Exception as e:
            logger.exception(f"Error in count_tokens request: {e}")
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    def _request_id_prefix(self) -> str:
        return "msg_"

    # DESIGN NOTE: Some models that expect thinking blocks re-wrapped as <think>...</think> in conversation history
    # and sent back as part of "interleaved thinking".
    # Unfortunately there is no signifier in the model config that we can track to identify these models.
    # The list below is based on model author's claims and checks for known tokenizer names.
    INTERLEAVED_THINKING_MODELS = frozenset(
        {
            "glm45",
            "kimi_k2",
            "minimax",
        }
    )

    def _model_uses_interleaved_thinking(self) -> bool:
        """Check if the model expects thinking blocks wrapped in <think> tags in history."""
        if not self.reasoning_parser:
            return False
        return self.reasoning_parser.lower() in self.INTERLEAVED_THINKING_MODELS

    def _rewrap_thinking_blocks_to_dict(
        self, message: AnthropicMessage
    ) -> Dict[str, Any]:
        """Convert message to dict, rewrapping thinking blocks for interleaved thinking models.

        These models expect thinking wrapped in <think> tags in conversation history.
        Only applies to assistant messages with structured content blocks.
        """
        match message.content:
            case str():
                return message.model_dump()
            case list() if message.role != "assistant":
                return message.model_dump()

        new_content_parts = []
        for block in message.content:
            match block:
                case ThinkingContentBlock():
                    new_content_parts.append(
                        {"type": "text", "text": f"<think>{block.thinking}</think>"}
                    )
                case _:
                    new_content_parts.append(block.model_dump())

        return {"role": message.role, "content": new_content_parts}

    def _parse_finish_reason(
        self, meta_info: Dict[str, Any]
    ) -> Optional[AnthropicFinishReason]:
        """Parse finish_reason from meta_info into AnthropicFinishReason model"""
        finish_reason_data = meta_info.get("finish_reason")
        if not finish_reason_data:
            return None

        if isinstance(finish_reason_data, dict):
            try:
                return AnthropicFinishReason(**finish_reason_data)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse finish_reason: {e}. Data: {finish_reason_data}"
                )
        return None

    def _map_finish_reason_to_stop_reason(
        self, finish_reason: Optional[AnthropicFinishReason]
    ) -> str:
        """Map internal finish reason to Anthropic stop_reason"""
        if not finish_reason:
            return "end_turn"

        if finish_reason.type == "length":
            return "max_tokens"
        elif finish_reason.type == "stop":
            return "stop_sequence"
        elif finish_reason.type == "abort":
            # Map specific abort types to Anthropic stop reasons
            # Could check finish_reason.err_type for:
            # - "context_overflow" -> "model_context_window_exceeded"
            # - "refusal" -> "refusal"
            # For now, treat aborts as errors rather than setting stop_reason
            logger.warning(f"Request aborted: {finish_reason.message}")
            return "end_turn"

        return "end_turn"

    def _convert_to_internal_request(
        self, request: AnthropicMessagesRequest
    ) -> GenerateReqInput:
        """Convert Anthropic request to SGLang internal format"""

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        processed_messages = self._process_messages(request, is_multimodal)

        sampling_params = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
        }

        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k

        if request.stop_sequences:
            sampling_params["stop"] = request.stop_sequences

        # Apply tool_call_constraint if present (for forced tool use)
        if processed_messages.tool_call_constraint:
            constraint_type, constraint_value = processed_messages.tool_call_constraint
            if constraint_type == "json_schema":
                sampling_params["json_schema"] = json.dumps(constraint_value)

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=== RAW ANTHROPIC REQUEST ===")
            logger.debug(f"Model: {request.model}")
            logger.debug(f"System: {request.system}")
            logger.debug(
                f"Tools: {[t.name for t in request.tools] if request.tools else None}"
            )
            logger.debug(f"Tool choice: {request.tool_choice}")
            for i, msg in enumerate(request.messages):
                logger.debug(f"Message[{i}] role={msg.role}:")
                match msg.content:
                    case str():
                        logger.debug(
                            f"  content (str): {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}"
                        )
                    case list():
                        for j, block in enumerate(msg.content):
                            match block:
                                case TextContentBlock():
                                    logger.debug(
                                        f"  block[{j}] type=text: {block.text[:100]}{'...' if len(block.text) > 100 else ''}"
                                    )
                                case ToolUseContentBlock():
                                    logger.debug(
                                        f"  block[{j}] type=tool_use: id={block.id} name={block.name} input={block.input}"
                                    )
                                case ToolResultContentBlock():
                                    logger.debug(
                                        f"  block[{j}] type=tool_result: tool_use_id={block.tool_use_id} content={block.content}"
                                    )
                                case ThinkingContentBlock():
                                    logger.debug(
                                        f"  block[{j}] type=thinking: {block.thinking[:100]}{'...' if len(block.thinking) > 100 else ''}"
                                    )
                                case _:
                                    logger.debug(
                                        f"  block[{j}] type={block.type}: {block}"
                                    )
            logger.debug("=== END RAW ANTHROPIC REQUEST ===")

        tools = None
        tool_call_constraint: Optional[ToolCallConstraint] = None
        if request.tools:
            converted = self._convert_anthropic_tools_to_openai(
                request.tools, request.tool_choice
            )
            tools = converted.get("tools")
            logger.debug(
                f"Converted {len(request.tools)} Anthropic tools to OpenAI format: {[t['function']['name'] for t in tools]}"
            )

            # Calculate tool_call_constraint for forced tool use
            if request.tool_choice and request.tool_choice.type in ("any", "tool"):
                openai_tools = self._convert_anthropic_tools_to_openai_objects(
                    request.tools
                )
                if request.tool_choice.type == "any":
                    # "any" means required - must use some tool
                    json_schema = get_json_schema_constraint(openai_tools, "required")
                    if json_schema:
                        tool_call_constraint = ("json_schema", json_schema)
                elif request.tool_choice.type == "tool" and request.tool_choice.name:
                    # Specific tool required
                    openai_tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=request.tool_choice.name),
                    )
                    json_schema = get_json_schema_constraint(
                        openai_tools, openai_tool_choice
                    )
                    if json_schema:
                        tool_call_constraint = ("json_schema", json_schema)

        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
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

        if request.system:
            match request.system:
                case str():
                    system_content = request.system
                case list():
                    system_content = "\n".join(block.text for block in request.system)
            openai_compatible_messages.append(
                {"role": "system", "content": system_content}
            )

        has_tool_blocks = any(
            not isinstance(msg.content, str)
            and any(
                isinstance(block, (ToolUseContentBlock, ToolResultContentBlock))
                for block in msg.content
            )
            for msg in request.messages
        )

        if has_tool_blocks:
            converted_messages = self._convert_anthropic_messages_to_openai(
                request.messages
            )
            openai_compatible_messages.extend(converted_messages)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "=== CONVERTED TO OPENAI FORMAT (tool blocks detected) ==="
                )
                for i, msg in enumerate(converted_messages):
                    role = msg.get("role")
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                    tool_call_id = msg.get("tool_call_id")
                    if role == "tool":
                        logger.debug(
                            f"  [{i}] role=tool tool_call_id={tool_call_id} content={content}"
                        )
                    elif tool_calls:
                        logger.debug(
                            f"  [{i}] role={role} content={content[:100] if content else None}... tool_calls={tool_calls}"
                        )
                    else:
                        content_preview = str(content)[:100] if content else None
                        logger.debug(f"  [{i}] role={role} content={content_preview}")
                logger.debug("=== END CONVERTED MESSAGES ===")
        else:
            for message in request.messages:
                # For interleaved thinking models, re-wrap thinking blocks back to <think>...</think>
                if self._model_uses_interleaved_thinking():
                    msg_dict = self._rewrap_thinking_blocks_to_dict(message)
                else:
                    msg_dict = message.model_dump()

                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )
                openai_compatible_messages.append(processed_msg)

        assistant_prefix = None
        if (
            openai_compatible_messages
            and openai_compatible_messages[-1]["role"] == "assistant"
        ):
            last_msg = openai_compatible_messages[-1]
            # Don't extract prefix if the message has tool_calls - those need to stay in the conversation
            if "tool_calls" not in last_msg or not last_msg.get("tool_calls"):
                assistant_prefix = last_msg.get("content")
                openai_compatible_messages = openai_compatible_messages[:-1]

        prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
            openai_compatible_messages,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )

        if assistant_prefix:
            if isinstance(assistant_prefix, list):
                prefix_text = ""
                for block in assistant_prefix:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and "text" in block:
                            prefix_text += block["text"]
                        elif block.get("type") == "thinking" and "thinking" in block:
                            # For interleaved thinking models, wrap thinking in <think> tags
                            # For other models, include thinking as raw text
                            thinking_content = block["thinking"]
                            if self._model_uses_interleaved_thinking():
                                prefix_text += f"<think>{thinking_content}</think>"
                            else:
                                prefix_text += thinking_content
                assistant_prefix = prefix_text

            if assistant_prefix:
                encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
                if (
                    encoded
                    and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id
                ):
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

        if request.system:
            match request.system:
                case str():
                    system_content = request.system
                case list():
                    system_content = "\n".join(block.text for block in request.system)
            conv.set_system_message(system_content)

        if request.messages and request.messages[-1].role == "assistant":
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            prompt = conv.get_prompt()
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
        stop = list(request.stop_sequences) if request.stop_sequences else []

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
        return TextContentBlock(text=text)

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
        tool_choice: Optional[AnthropicToolChoice] = None,
    ) -> Dict[str, Any]:
        """Convert Anthropic tools to OpenAI function format"""
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": tool.input_schema,
                },
            }
            openai_tools.append(openai_tool)

        result = {"tools": openai_tools}

        if tool_choice:
            match tool_choice.type:
                case "auto":
                    result["tool_choice"] = "auto"
                case "any":
                    result["tool_choice"] = "required"
                case "tool":
                    result["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice.name},
                    }

        return result

    def _convert_anthropic_tools_to_openai_objects(
        self,
        tools: List[AnthropicTool],
    ) -> List[Tool]:
        """Convert Anthropic tools to OpenAI Tool objects (Pydantic models) for FunctionCallParser"""
        return [
            Tool(
                type="function",
                function=Function(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.input_schema,
                ),
            )
            for tool in tools
        ]

    def _convert_anthropic_messages_to_openai(
        self, messages: List[AnthropicMessage]
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic message format to OpenAI format for apply_chat_template.

        Anthropic format:
        - Assistant messages have content blocks including tool_use blocks
        - User messages can have tool_result content blocks

        OpenAI format:
        - Assistant messages have tool_calls array (not content blocks)
        - Tool results are separate messages with role="tool"
        """
        openai_messages = []

        for message in messages:
            match message.content:
                case str():
                    openai_messages.append(
                        {"role": message.role, "content": message.content}
                    )
                    continue
                case list():
                    pass  # Process below

            if message.role == "assistant":
                text_parts = []
                tool_calls = []

                for block in message.content:
                    match block:
                        case TextContentBlock():
                            text_parts.append(block.text)
                        case ThinkingContentBlock():
                            if self._model_uses_interleaved_thinking():
                                text_parts.append(f"<think>{block.thinking}</think>")
                            else:
                                text_parts.append(block.thinking)
                        case ToolUseContentBlock():
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": block.input,
                                    },
                                }
                            )

                assistant_msg = {"role": "assistant"}
                content = "".join(text_parts)
                if content:
                    assistant_msg["content"] = content
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                openai_messages.append(assistant_msg)

            elif message.role == "user":
                text_parts = []
                tool_results = []
                other_content = []

                for block in message.content:
                    match block:
                        case TextContentBlock():
                            text_parts.append(block.text)
                        case ToolResultContentBlock():
                            tool_result_content = block.content
                            if isinstance(tool_result_content, list):
                                parts = []
                                for item in tool_result_content:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        if "text" not in item:
                                            raise ValueError(
                                                "text item in tool_result missing 'text' field"
                                            )
                                        parts.append(item["text"])
                                tool_result_content = " ".join(parts)
                            tool_results.append(
                                {
                                    "tool_call_id": block.tool_use_id,
                                    "content": tool_result_content,
                                }
                            )
                        case _:
                            other_content.append(block.model_dump())

                if text_parts:
                    openai_messages.append(
                        {"role": "user", "content": "".join(text_parts)}
                    )
                elif other_content and not tool_results:
                    openai_messages.append({"role": "user", "content": other_content})
                elif not tool_results:
                    openai_messages.append({"role": "user", "content": ""})

                for tool_result in tool_results:
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"],
                        }
                    )

        return openai_messages

    def _convert_function_calls_to_anthropic(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[AnthropicContentBlock]:
        """Convert OpenAI function calls to Anthropic tool_use blocks"""
        content_blocks = []

        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                arguments_str = function.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse tool call arguments as JSON: {e}. "
                        f"Raw arguments: {arguments_str[:200]}"
                    )

                tool_use_block = ToolUseContentBlock(
                    id=tool_call.get("id"),
                    name=function.get("name"),
                    input=arguments,
                )
                content_blocks.append(tool_use_block)

        return content_blocks

    def _create_stream_event(self, event_type: str, **kwargs) -> AnthropicStreamEvent:
        """Create streaming event"""
        return AnthropicStreamEvent(type=event_type, **kwargs)

    async def _handle_streaming_request(
        self,
        internal_request: GenerateReqInput,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response in Anthropic format"""

        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"

        stream_buffers = {}
        reasoning_parser_dict = {}
        tool_call_parser_dict = {}

        start_message = AnthropicMessagesResponse(
            id=request_id,
            content=[],
            model=anthropic_request.model,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )

        start_event = self._create_stream_event("message_start", message=start_message)
        yield f"event: message_start\ndata: {orjson.dumps(start_event.model_dump()).decode()}\n\n"

        thinking_started = False
        thinking_accumulated = ""
        text_started = False
        accumulated_text = ""
        tool_use_started = False
        tool_use_blocks = []
        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"
        final_meta_info = {}
        current_content_block_index = 0

        try:
            async for content in self.tokenizer_manager.generate_request(
                internal_request, raw_request
            ):
                index = content.get("index", 0)
                if "text" in content:
                    stream_buffer = stream_buffers.get(index, "")
                    text_delta = content["text"][len(stream_buffer) :]
                    stream_buffers[index] = stream_buffer + text_delta

                    thinking_text = None
                    if self.reasoning_parser:
                        if index not in reasoning_parser_dict:
                            is_force_reasoning = (
                                self.template_manager.force_reasoning
                                if hasattr(self.template_manager, "force_reasoning")
                                else False
                            )
                            reasoning_parser_dict[index] = ReasoningParser(
                                model_type=self.reasoning_parser,
                                stream_reasoning=True,
                                force_reasoning=is_force_reasoning,
                            )
                        thinking_text, text_delta = reasoning_parser_dict[
                            index
                        ].parse_stream_chunk(text_delta)

                    tool_calls_from_stream = []
                    if self.tool_call_parser and anthropic_request.tools:
                        if index not in tool_call_parser_dict:
                            openai_tools = (
                                self._convert_anthropic_tools_to_openai_objects(
                                    anthropic_request.tools
                                )
                            )
                            tool_call_parser_dict[index] = FunctionCallParser(
                                openai_tools, self.tool_call_parser
                            )

                        text_delta, tool_calls_from_stream = tool_call_parser_dict[
                            index
                        ].parse_stream_chunk(text_delta)

                    for call_item in tool_calls_from_stream:
                        # Close active blocks before starting tool_use
                        if (thinking_started or text_started) and not tool_use_started:
                            stop_event = self._create_stream_event(
                                "content_block_stop", index=current_content_block_index
                            )
                            yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                            current_content_block_index += 1
                            thinking_started = False
                            text_started = False

                        if call_item.name:
                            if tool_use_started:
                                stop_event = self._create_stream_event(
                                    "content_block_stop",
                                    index=current_content_block_index,
                                )
                                yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                                current_content_block_index += 1

                            tool_use_block = ToolUseContentBlock(
                                id=f"toolu_{uuid.uuid4().hex[:24]}",
                                name=call_item.name,
                                input={},
                            )
                            tool_use_blocks.append(tool_use_block)

                            content_start_event = self._create_stream_event(
                                "content_block_start",
                                index=current_content_block_index,
                                content_block=tool_use_block,
                            )
                            yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event.model_dump()).decode()}\n\n"
                            tool_use_started = True

                        if call_item.parameters:
                            delta = AnthropicDelta(
                                type="input_json_delta",
                                partial_json=call_item.parameters,
                            )
                            delta_event = self._create_stream_event(
                                "content_block_delta",
                                index=current_content_block_index,
                                delta=delta,
                            )
                            yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event.model_dump()).decode()}\n\n"

                    if thinking_text:
                        if tool_use_started and not thinking_started:
                            stop_event = self._create_stream_event(
                                "content_block_stop", index=current_content_block_index
                            )
                            yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                            current_content_block_index += 1
                            tool_use_started = False

                        if not thinking_started:
                            thinking_block = ThinkingContentBlock(
                                thinking="", signature="N/A"
                            )
                            content_start_event = self._create_stream_event(
                                "content_block_start",
                                index=current_content_block_index,
                                content_block=thinking_block,
                            )
                            yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event.model_dump()).decode()}\n\n"
                            thinking_started = True

                        thinking_accumulated += thinking_text
                        delta = AnthropicDelta(
                            type="thinking_delta", thinking=thinking_text
                        )
                        delta_event = self._create_stream_event(
                            "content_block_delta",
                            index=current_content_block_index,
                            delta=delta,
                        )
                        yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event.model_dump()).decode()}\n\n"

                    if thinking_started and text_delta and not thinking_text:
                        stop_event = self._create_stream_event(
                            "content_block_stop", index=current_content_block_index
                        )
                        yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                        current_content_block_index += 1
                        thinking_started = False
                        text_started = False

                    if text_delta:
                        if tool_use_started and not text_started:
                            stop_event = self._create_stream_event(
                                "content_block_stop", index=current_content_block_index
                            )
                            yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                            current_content_block_index += 1
                            tool_use_started = False

                        if not text_started:
                            text_block = TextContentBlock(text="")
                            content_start_event = self._create_stream_event(
                                "content_block_start",
                                index=current_content_block_index,
                                content_block=text_block,
                            )
                            yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event.model_dump()).decode()}\n\n"
                            text_started = True

                        accumulated_text += text_delta
                        delta = AnthropicDelta(type="text_delta", text=text_delta)
                        delta_event = self._create_stream_event(
                            "content_block_delta",
                            index=current_content_block_index,
                            delta=delta,
                        )
                        yield f"event: content_block_delta\ndata: {orjson.dumps(delta_event.model_dump()).decode()}\n\n"

                if "tool_calls" in content or "function_calls" in content:
                    tool_calls = content.get("tool_calls") or content.get(
                        "function_calls", []
                    )
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            arguments_str = function.get("arguments", "{}")
                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to parse tool call arguments as JSON: {e}. "
                                    f"Raw arguments: {arguments_str[:200]}"
                                )
                                raise

                            if thinking_started or text_started:
                                stop_event = self._create_stream_event(
                                    "content_block_stop",
                                    index=current_content_block_index,
                                )
                                yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                                current_content_block_index += 1
                                thinking_started = False
                                text_started = False

                            tool_use_block = ToolUseContentBlock(
                                id=tool_call.get("id"),
                                name=function.get("name"),
                                input=arguments,
                            )
                            tool_use_blocks.append(tool_use_block)

                            content_start_event = self._create_stream_event(
                                "content_block_start",
                                index=current_content_block_index,
                                content_block=tool_use_block,
                            )
                            yield f"event: content_block_start\ndata: {orjson.dumps(content_start_event.model_dump()).decode()}\n\n"

                            stop_event = self._create_stream_event(
                                "content_block_stop", index=current_content_block_index
                            )
                            yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"
                            current_content_block_index += 1
                            stop_reason = "tool_use"

                if "meta_info" in content:
                    meta = content["meta_info"]
                    final_meta_info = meta
                    input_tokens = meta.get("prompt_tokens", 0)
                    output_tokens = meta.get("completion_tokens", 0)

        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            logger.error(f"Traceback: {get_exception_traceback()}")
            error_event = self._create_stream_event(
                "error",
                error=AnthropicError(type="internal_server_error", message=str(e)),
            )
            yield f"event: error\ndata: {orjson.dumps(error_event.model_dump()).decode()}\n\n"
            return

        if thinking_started or text_started or tool_use_started:
            stop_event = self._create_stream_event(
                "content_block_stop", index=current_content_block_index
            )
            yield f"event: content_block_stop\ndata: {orjson.dumps(stop_event.model_dump()).decode()}\n\n"

        if tool_use_blocks:
            stop_reason = "tool_use"
        else:
            finish_reason = self._parse_finish_reason(final_meta_info)
            stop_reason = self._map_finish_reason_to_stop_reason(finish_reason)

        final_usage = AnthropicUsage(
            input_tokens=input_tokens, output_tokens=output_tokens
        )

        final_content_blocks = []
        if thinking_accumulated:
            final_content_blocks.append(
                ThinkingContentBlock(thinking=thinking_accumulated, signature="N/A")
            )
        if accumulated_text:
            final_content_blocks.append(
                self._create_anthropic_content_block(accumulated_text)
            )
        if tool_use_blocks:
            final_content_blocks.extend(tool_use_blocks)

        final_message = AnthropicMessagesResponse(
            id=request_id,
            content=final_content_blocks,
            model=anthropic_request.model,
            stop_reason=stop_reason,
            usage=final_usage,
        )

        stop_message_event = self._create_stream_event(
            "message_stop", message=final_message
        )
        yield f"event: message_stop\ndata: {orjson.dumps(stop_message_event.model_dump()).decode()}\n\n"

    async def _handle_non_streaming_request(
        self,
        internal_request: GenerateReqInput,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AnthropicMessagesResponse:

        response_generator = self.tokenizer_manager.generate_request(
            internal_request, raw_request
        )

        full_response = await response_generator.__anext__()

        content = full_response.get("text", "")
        meta_info = full_response.get("meta_info", {})
        input_tokens = meta_info.get("prompt_tokens", 0)
        output_tokens = meta_info.get("completion_tokens", 0)

        content_blocks = self._parse_content_blocks(
            content, full_response, anthropic_request
        )

        has_tool_calls = any(block.type == "tool_use" for block in content_blocks)
        if has_tool_calls:
            stop_reason = "tool_use"
        else:
            finish_reason = self._parse_finish_reason(meta_info)
            stop_reason = self._map_finish_reason_to_stop_reason(finish_reason)

        response = AnthropicMessagesResponse(
            content=content_blocks,
            model=anthropic_request.model,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )

        return response

    def _parse_content_blocks(
        self,
        content: str,
        full_response: Dict[str, Any],
        anthropic_request: AnthropicMessagesRequest,
    ) -> List[AnthropicContentBlock]:
        """Parse content into Anthropic content blocks (thinking, text, tool_use)."""
        logger.debug(f"Parsing {len(content)} chars of model output")

        content_blocks = []
        remaining_text = content

        tool_calls = []
        if self.tool_call_parser and anthropic_request.tools:
            try:
                openai_tools = self._convert_anthropic_tools_to_openai_objects(
                    anthropic_request.tools
                )
                parser = FunctionCallParser(openai_tools, self.tool_call_parser)
                remaining_text, call_info_list = parser.parse_non_stream(content)

                logger.debug(
                    f"Extracted {len(call_info_list)} tool calls, {len(remaining_text)} chars remaining"
                )

                for call_info in call_info_list:
                    tool_calls.append(
                        ToolUseContentBlock(
                            id=f"toolu_{uuid.uuid4().hex[:24]}",
                            name=call_info.name,
                            input=(
                                json.loads(call_info.parameters)
                                if isinstance(call_info.parameters, str)
                                else call_info.parameters
                            ),
                        )
                    )
            except Exception as e:
                logger.error(f"Error parsing tool calls: {e}")
                logger.error(f"Traceback: {get_exception_traceback()}")
                raise

        thinking_text = ""
        if self.reasoning_parser:
            try:
                parser = ReasoningParser(
                    model_type=self.reasoning_parser,
                    stream_reasoning=False,
                    force_reasoning=False,
                )
                thinking_text, remaining_text = parser.parse_non_stream(remaining_text)
                if thinking_text:
                    logger.debug(
                        f"Extracted {len(thinking_text)} chars thinking, {len(remaining_text)} chars remaining"
                    )
            except Exception as e:
                logger.error(f"Error parsing reasoning: {e}")
                raise

        # Assemble blocks: thinking → text → tool_use
        if thinking_text:
            content_blocks.append(
                ThinkingContentBlock(thinking=thinking_text, signature="N/A")
            )

        if remaining_text and remaining_text.strip():
            content_blocks.append(TextContentBlock(text=remaining_text.strip()))

        content_blocks.extend(tool_calls)

        if not content_blocks and content:
            content_blocks.append(TextContentBlock(text=content))

        logger.debug(
            f"Content blocks: {[(b.type, getattr(b, 'name', None)) for b in content_blocks]}"
        )
        return content_blocks

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        error = AnthropicErrorResponse(
            type="error", error=AnthropicError(type=err_type, message=message)
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
    ) -> str:
        """Create a streaming error response"""
        error = AnthropicErrorResponse(
            type="error", error=AnthropicError(type=err_type, message=message)
        )
        return json.dumps({"error": error.model_dump()})
