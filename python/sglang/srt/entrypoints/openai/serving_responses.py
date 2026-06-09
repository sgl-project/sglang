# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from contextlib import AsyncExitStack
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai_harmony import Message as OpenAIMessage

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    Function,
    LogProbs,
    OutputTokenUsageInfo,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsageInfo,
    Tool,
    ToolChoice,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.entrypoints.openai.utils import (
    parse_tool_calls_from_content,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import _is_complete_json
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _normalize_responses_input(
    input_items: list,
) -> list[ChatCompletionMessageParam]:
    """Normalize Responses API input messages to Chat Completions format.

    Converts Responses API items (message, reasoning, function_call, function_call_output)
    to Chat Completions messages format, merging related assistant items.
    """
    messages: list[ChatCompletionMessageParam] = []
    pending_assistant: Optional[dict] = None

    def _flush_pending_assistant():
        """Add pending assistant message to messages if exists."""
        nonlocal pending_assistant
        if pending_assistant is not None:
            # Clean up empty fields
            if not pending_assistant.get("content"):
                pending_assistant["content"] = None
            if not pending_assistant.get("tool_calls"):
                del pending_assistant["tool_calls"]
            messages.append(pending_assistant)  # type: ignore[arg-type]
            pending_assistant = None

    for raw_item in input_items:
        # Convert Pydantic models to dicts for uniform handling
        item = raw_item
        if hasattr(raw_item, "model_dump"):
            item = raw_item.model_dump()

        if not isinstance(item, dict):
            messages.append(raw_item)  # type: ignore[arg-type]
            continue

        item_type = item.get("type", "message")
        role = item.get("role")

        if item_type == "message" and role == "user":
            _flush_pending_assistant()
            messages.append(
                {
                    "role": "user",
                    "content": _normalize_content(item.get("content")),
                }
            )

        elif item_type == "reasoning":
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": [],
                }
            # Extract reasoning text from summary or content
            reasoning_text = ""
            summary = item.get("summary", [])
            if summary:
                for s in summary:
                    if hasattr(s, "model_dump"):
                        s = s.model_dump()
                    if isinstance(s, dict) and s.get("type") == "summary_text":
                        reasoning_text = s.get("text", "")
                        break
            if not reasoning_text:
                content = item.get("content", [])
                if content:
                    for c in content:
                        if hasattr(c, "model_dump"):
                            c = c.model_dump()
                        if isinstance(c, dict) and c.get("type") == "reasoning_text":
                            reasoning_text = c.get("text", "")
                            break
            pending_assistant["reasoning_content"] = reasoning_text

        elif item_type == "function_call":
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": [],
                }
            if "tool_calls" not in pending_assistant:
                pending_assistant["tool_calls"] = []
            pending_assistant["tool_calls"].append(
                {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )

        elif item_type == "message" and role == "assistant":
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": None,
                    "tool_calls": [],
                }
            pending_assistant["content"] = _normalize_content(item.get("content"))

        elif item_type == "function_call_output":
            _flush_pending_assistant()
            # Find the corresponding function_call to get the name
            call_id = item.get("call_id", "")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _normalize_content(item.get("output", "")),
                }
            )

        else:
            # Pass through other items (e.g., system messages)
            _flush_pending_assistant()
            normalized_item = dict(item)
            if "content" in normalized_item:
                normalized_item["content"] = _normalize_content(
                    normalized_item["content"]
                )
            messages.append(normalized_item)  # type: ignore[arg-type]

    _flush_pending_assistant()
    return messages


def _normalize_content(content) -> Optional[Union[str, list]]:
    """Normalize content: convert input_text/output_text to text type."""
    if content is None:
        return None
    if isinstance(content, str):
        return content

    if hasattr(content, "__iter__") and not isinstance(content, list):
        try:
            content = list(content)
        except Exception as e:
            logger.warning(f"Failed to convert content iterator: {e}")
            return str(content)

    if isinstance(content, list):
        normalized = []
        for part in content:
            if hasattr(part, "model_dump"):
                part = part.model_dump()
            normalized.append(_normalize_content_part(part))
        return normalized
    return content


def _normalize_content_part(part: dict) -> dict:
    """Convert a Responses API content part to Chat Completions format."""
    if isinstance(part, dict):
        part_type = part.get("type")
        if part_type in ("input_text", "output_text"):
            return {"type": "text", "text": part.get("text", "")}
        elif part_type == "input_image":
            return {
                "type": "image_url",
                "image_url": {
                    "url": part.get("image_url", ""),
                    "detail": part.get("detail", "auto"),
                },
            }
        elif part_type == "input_audio":
            return {"type": "audio_url", "audio_url": {"url": part.get("data", "")}}
    return part


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        tool_server: Optional[ToolServer] = None,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

        # Get default sampling params from model config if available
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if self.default_sampling_params:
            logger.info(
                f"OpenAIServingResponses init,  Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )

        self.supports_browsing = (
            tool_server.has_tool("browser") if tool_server else False
        )
        self.supports_code_interpreter = (
            tool_server.has_tool("python") if tool_server else False
        )
        self.tool_server = tool_server
        # Get from model config
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Response storage for background and retrieval operations
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # Message storage for conversation continuity
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.msg_store: dict[
            str, Union[list[ChatCompletionMessageParam], list["OpenAIMessage"]]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

    def _get_function_tools(self, request: ResponsesRequest) -> list[Tool]:
        """Extract function tools from request and convert to Tool format."""
        function_tools = []
        for tool in request.tools:
            if tool.type == "function":
                if not tool.name:
                    logger.warning(
                        f"Skipping function tool without function name definition: {tool}"
                    )
                    continue
                function_tools.append(
                    Tool(
                        type="function",
                        function=Function(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                            strict=tool.strict or False,
                        ),
                    )
                )
        return function_tools

    # error helpers dedicated for v1/responses
    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code,
        }
        return ORJSONResponse(content={"error": nested_error}, status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": status_code,
                }
            }
        )

    def _request_id_prefix(self) -> str:
        return "resp_"

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ORJSONResponse]:
        # Validate model
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")

        # FIXME: If the engine is dead, raise an error
        # This is required for the streaming case

        # Handle the previous response ID
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            model_name = request.model
            tokenizer = self.tokenizer_manager.tokenizer

            if self.use_harmony:
                messages, request_prompts, engine_prompts, processed_messages = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                messages, request_prompts, engine_prompts, processed_messages = (
                    await self._make_request(request, prev_response, tokenizer)
                )

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ["web_search_preview", "code_interpreter"]
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and "
                "streaming mode"
            )

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        tool_list = []
        if self.use_harmony:
            if self.supports_browsing:
                tool_list.append("browser")
            if self.supports_code_interpreter:
                tool_list.append("python")
        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.get_tool_session(tool_name)
                        )
                        for tool_name in tool_list
                    }
                    tool_sessions = {}
                    for tool_name in tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(tool_list) == 0
                    tool_sessions = {}
                for i, engine_prompt in enumerate(engine_prompts):
                    # Calculate default max tokens from context length minus prompt length
                    if isinstance(engine_prompt, list):
                        prompt_length = len(engine_prompt)
                    elif isinstance(engine_prompt, str):
                        prompt_length = len(tokenizer.encode(engine_prompt))
                    else:
                        prompt_length = 0

                    context_len = (
                        self.tokenizer_manager.model_config.context_len
                        if hasattr(self.tokenizer_manager.model_config, "context_len")
                        else 4096
                    )
                    # Account for reserved tokens (e.g., EAGLE speculative decoding slots)
                    # that the tokenizer_manager adds during validation
                    num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                    default_max_tokens = max(
                        context_len - prompt_length - num_reserved_tokens, 512
                    )  # Ensure minimum 512 tokens
                    sampling_params = request.to_sampling_params(
                        default_max_tokens=default_max_tokens,
                        default_params=self.default_sampling_params,
                        stop=(
                            processed_messages.stop
                            if processed_messages
                            else request.stop
                        ),
                        tool_call_constraint=(
                            processed_messages.tool_call_constraint
                            if processed_messages
                            else None
                        ),
                    )

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    if isinstance(engine_prompt, str):
                        prompt_kwargs = {"text": engine_prompt}
                    else:
                        prompt_kwargs = {"input_ids": engine_prompt}

                    should_return_logprobs = self._should_return_logprobs(request)
                    # Extract routed_dp_rank from header (has higher priority than body)
                    effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
                        raw_request, request.routed_dp_rank
                    )

                    # Create GenerateReqInput for SGLang
                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        sampling_params=sampling_params,
                        stream=request.stream,
                        rid=request.request_id,
                        extra_key=self._compute_extra_key(request),
                        background=request.background,
                        require_reasoning=self._get_reasoning_from_request(request),
                        bootstrap_host=request.bootstrap_host,
                        bootstrap_port=request.bootstrap_port,
                        bootstrap_room=request.bootstrap_room,
                        routed_dp_rank=effective_routed_dp_rank,
                        disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
                        image_data=processed_messages.image_data,
                        video_data=processed_messages.video_data,
                        audio_data=processed_messages.audio_data,
                        modalities=processed_messages.modalities,
                        priority=request.priority,
                        routing_key=self.extract_routing_key(raw_request),
                        custom_labels=self.extract_custom_labels(raw_request),
                        return_logprob=should_return_logprobs,
                        logprob_start_len=-1,
                        top_logprobs_num=(
                            request.top_logprobs if should_return_logprobs else 0
                        ),
                        return_text_in_logprobs=True,
                    )

                    generator = self._generate_with_builtin_tools(
                        request.request_id,
                        request_prompts[i],
                        adapted_request,
                        sampling_params,
                        context,
                        raw_request=raw_request,
                        priority=request.priority,
                    )
                    generators.append(generator)
            except ValueError as e:
                return self.create_error_response(str(e))

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages
            if request.store:
                self.msg_store[request.request_id] = messages

            if request.background:
                created_time = int(time.time())
                response = ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="queued",
                    usage=None,
                )
                async with self.response_store_lock:
                    self.response_store[response.id] = response

                # Run the request in the background
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{response.id}",
                )

                # For cleanup
                self.background_tasks[response.id] = task
                task.add_done_callback(
                    lambda _: self.background_tasks.pop(response.id, None)
                )
                return response

            if request.stream:
                return self.responses_stream_generator(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                )
            try:
                result: Union[ORJSONResponse, ResponsesResponse] = (
                    await self.responses_full_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                    )
                )
                return result
            except Exception as e:
                return self.create_error_response(str(e))
        return self.create_error_response("Unknown error")

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        # Construct the input messages
        messages = self._construct_input_messages(request, prev_response)
        function_tools = self._get_function_tools(request)

        # Follow SGLang's pattern: create a ChatCompletionRequest and process messages
        try:
            # Convert ResponsesRequest to ChatCompletionRequest for processing
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                stream=request.stream,
                tools=function_tools,
                tool_choice=request.tool_choice if function_tools else "none",
                parallel_tool_calls=request.parallel_tool_calls,
                chat_template_kwargs=request.chat_template_kwargs,
                reasoning_effort=(
                    request.reasoning.effort if request.reasoning else None
                ),
                stop=request.stop,
            )

            # Follow SGLang's _process_messages pattern
            is_multimodal = self.tokenizer_manager.model_config.is_multimodal
            processed_messages = self._process_messages(chat_request, is_multimodal)

            # Extract the results
            if is_multimodal:
                request_prompts = [processed_messages.prompt]
                engine_prompts = [processed_messages.prompt]
            else:
                request_prompts = [processed_messages.prompt_ids]
                engine_prompts = [processed_messages.prompt_ids]

        except Exception as e:
            logger.warning(f"Chat processing failed, using fallback: {e}")
            # Fallback to simple encoding
            prompt_text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    role = msg.role if hasattr(msg, "role") else "user"
                    content = msg.content if hasattr(msg, "content") else ""
                prompt_text += f"{role}: {content}\n"
            prompt_ids = tokenizer.encode(prompt_text)
            request_prompts = [prompt_ids]
            engine_prompts = [prompt_ids]
            processed_messages = None

        return messages, request_prompts, engine_prompts, processed_messages

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in " "response API"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = prompt_token_ids
        return messages, [prompt_token_ids], [engine_prompt], None

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # TODO: these are all 0 for now!
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            logprobs = None
            if self._should_return_logprobs(request):
                logprobs = self._build_response_logprobs(final_res)

            output = self._make_response_output_items(
                request, final_res["text"], logprobs, tokenizer
            )

            num_reasoning_tokens = 0
            # Calculate usage from actual output
            if "meta_info" in final_res:
                meta_info = final_res.get("meta_info", {})
                num_prompt_tokens = meta_info.get("prompt_tokens", 0)
                num_generated_tokens = meta_info.get("completion_tokens", 0)
                num_cached_tokens = meta_info.get("cached_tokens", 0)
                num_reasoning_tokens = meta_info.get("reasoning_tokens", 0)
            elif "prompt_token_ids" in final_res and "outputs" in final_res:
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res["prompt_token_ids"])
                    if final_res.get("prompt_token_ids")
                    else 0
                )
                num_generated_tokens = (
                    len(final_res["outputs"][0]["token_ids"])
                    if (
                        final_res.get("outputs")
                        and final_res["outputs"][0].get("token_ids")
                    )
                    else 0
                )
                num_cached_tokens = final_res.get("num_cached_tokens", 0)
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0

        usage = ResponseUsageInfo(
            input_tokens=num_prompt_tokens,
            output_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            output_tokens_details=OutputTokenUsageInfo(
                reasoning_tokens=num_reasoning_tokens
            ),
        )
        if self.enable_prompt_tokens_details:
            usage.input_tokens_details = PromptTokenUsageInfo(
                cached_tokens=num_cached_tokens
            )
        request_metadata.final_usage_info = usage

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status="completed",
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        return response

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        logprobs: Optional[list[dict]],
        tokenizer: Any,
    ):
        # Handle reasoning parsing if enabled
        if self.reasoning_parser:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            # Use standard reasoning parser (openai maps to T4Detector internally)
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=is_force_reasoning,
                request=request,
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=[
                    openai_responses_types.response_reasoning_item.Summary(
                        type="summary_text",
                        text=reasoning_content,
                    ),
                ],
                content=[],
                status="completed",
            )
            output_items.append(reasoning_item)

        if (
            request.tool_choice != "none"
            and request.tools
            and self.tool_call_parser
            and content
        ):
            # Check for function tools and parse tool calls
            function_tools = self._get_function_tools(request)
            if function_tools:
                remaining_text, tool_calls = self._parse_tool_calls(
                    content,
                    function_tools,
                    request.tool_choice,
                    request,
                )
                output_items.extend(tool_calls)
                content = remaining_text

        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=logprobs,
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        return output_items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )

        # Prepend the conversation history
        if prev_response is not None:
            # Add the previous messages
            prev_msg = self.msg_store[prev_response.id]
            messages.extend(prev_msg)

            # Add the previous output
            for output_item in prev_response.output:
                # NOTE: We skip the reasoning output of the previous response
                if isinstance(output_item, ResponseReasoningItem):
                    continue
                for content in output_item.content:
                    if content.type == "output_text":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content.text,
                            }
                        )

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            messages.extend(_normalize_responses_input(request.input))  # type: ignore
        return messages

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list["OpenAIMessage"]:
        messages: list["OpenAIMessage"] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                "web_search_preview" in tool_types and self.tool_server is not None
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.tool_server is not None
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                browser_description=(
                    self.tool_server.get_tool_description("browser")
                    if self.tool_server and enable_browser
                    else None
                ),
                python_description=(
                    self.tool_server.get_tool_description("python")
                    if self.tool_server and enable_code_interpreter
                    else None
                ),
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions, request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME: Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if (
                len(prev_msgs) > 0
                and hasattr(prev_msgs[-1], "channel")
                and prev_msgs[-1].channel == "final"
            ):  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if (
                        hasattr(prev_msgs[i], "channel")
                        and prev_msgs[i].channel == "final"
                    ):  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                del prev_msgs[prev_final_msg_idx + 1 :]
                for msg in recent_turn_msgs:
                    if (
                        hasattr(msg, "channel") and msg.channel != "analysis"
                    ):  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = copy(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *args,
        **kwargs,
    ):
        try:
            # Update the status to "in_progress"
            async with self.response_store_lock:
                stored_response = self.response_store.get(request.request_id)
                assert stored_response is not None
                stored_response.status = "in_progress"

            response = await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time,
                *args,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                return self.create_error_response(
                    err_type="invalid_request_error",
                    message="Cannot cancel a synchronous response.",
                )

            # Update the status to "cancelled"
            response.status = "cancelled"

        # The response_id is the same as the rid used when submitting the request
        self.tokenizer_manager.abort_request(rid=response_id)

        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_invalid_id_error(self, response_id: str):
        return self.create_error_response(
            message=(
                f"Invalid 'response_id': '{response_id}'. "
                "Expected an ID that begins with 'resp'."
            ),
            err_type="invalid_request_error",
            param="response_id",
        )

    def _make_not_found_error(self, response_id: str):
        return self.create_error_response(
            message=f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        current_content_index = 0
        current_output_index = 0
        current_item_id = f"item_{random_uuid()}"
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )
        # Initialize function call parser for streaming if function tools are present
        function_tools = self._get_function_tools(request)
        streaming_fc_parser: Optional[Union[FunctionCallParser, JsonArrayParser]] = None
        if (
            function_tools
            and self.tool_call_parser
            and not self.use_harmony
            and request.tool_choice != "none"
        ):
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                streaming_fc_parser = JsonArrayParser()
            else:
                streaming_fc_parser = FunctionCallParser(
                    function_tools, self.tool_call_parser
                )

        history_tool_calls_cnt = (
            self._get_history_tool_calls_cnt(request) if function_tools else 0
        )

        reasoning_parser: Optional[ReasoningParser] = None
        if self.reasoning_parser:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                force_reasoning=is_force_reasoning,
                request=request,
            )

        # Track state for non-Harmony streaming with function tools
        simple_ctx_prev_text = ""
        simple_ctx_sent_message_added = False
        simple_ctx_accumulated_text = ""
        simple_ctx_sent_reasoning_added = False
        simple_ctx_sent_reasoning_done = False
        simple_ctx_reasoning_text = ""
        n_prev_token = 0
        return_logprobs = self._should_return_logprobs(request)

        # store function call state
        tool_call_states = {}  # tool_index -> state dict

        async for ctx in result_generator:

            # Only process context objects that implement the `is_expecting_start()` method,
            # which indicates they support per-turn streaming (e.g., StreamingHarmonyContext).
            # Contexts without this method are skipped, as they do not represent a new turn
            # or are not compatible with per-turn handling in the /v1/responses endpoint.
            if not hasattr(ctx, "is_expecting_start"):
                if hasattr(ctx, "last_output") and ctx.last_output:
                    current_text = ""
                    if isinstance(ctx.last_output, dict):
                        current_text = ctx.last_output.get("text", "")
                    elif hasattr(ctx.last_output, "text"):
                        current_text = ctx.last_output.text

                    streaming_logprobs = None
                    if return_logprobs:
                        streaming_logprobs = (
                            self._process_streaming_logprobs_for_responses(
                                ctx.last_output,
                                n_prev_token,
                            )
                        )
                        meta_info = ctx.last_output.get("meta_info", {})
                        if meta_info.get("output_token_logprobs_length"):
                            n_prev_token = meta_info["output_token_logprobs_length"]

                    # Get new text since last iteration
                    if current_text and len(current_text) > len(simple_ctx_prev_text):
                        new_chunk = current_text[len(simple_ctx_prev_text) :]
                        simple_ctx_prev_text = current_text

                        if reasoning_parser:
                            reasoning_content, new_chunk = (
                                reasoning_parser.parse_stream_chunk(new_chunk)
                            )
                            if reasoning_content:
                                if not simple_ctx_sent_reasoning_added:
                                    simple_ctx_sent_reasoning_added = True
                                    yield _send_event(
                                        openai_responses_types.ResponseOutputItemAddedEvent(
                                            type="response.output_item.added",
                                            sequence_number=-1,
                                            output_index=current_output_index,
                                            item=openai_responses_types.ResponseReasoningItem(
                                                id=current_item_id,
                                                type="reasoning",
                                                status="in_progress",
                                                summary=[],
                                            ),
                                        )
                                    )
                                    yield _send_event(
                                        openai_responses_types.ResponseReasoningSummaryPartAddedEvent(
                                            type="response.reasoning_summary_part.added",
                                            sequence_number=-1,
                                            output_index=current_output_index,
                                            item_id=current_item_id,
                                            summary_index=0,
                                            part=openai_responses_types.response_reasoning_summary_part_added_event.Part(
                                                type="summary_text",
                                                text="",
                                            ),
                                        )
                                    )

                                yield _send_event(
                                    openai_responses_types.ResponseReasoningSummaryTextDeltaEvent(
                                        type="response.reasoning_summary_text.delta",
                                        sequence_number=-1,
                                        output_index=current_output_index,
                                        item_id=current_item_id,
                                        summary_index=0,
                                        delta=reasoning_content,
                                    )
                                )
                                simple_ctx_reasoning_text += reasoning_content

                            if (
                                new_chunk
                                and simple_ctx_sent_reasoning_added
                                and not simple_ctx_sent_reasoning_done
                            ):
                                yield _send_event(
                                    openai_responses_types.ResponseReasoningSummaryTextDoneEvent(
                                        type="response.reasoning_summary_text.done",
                                        item_id=current_item_id,
                                        sequence_number=-1,
                                        output_index=current_output_index,
                                        summary_index=0,
                                        text=simple_ctx_reasoning_text,
                                    )
                                )
                                yield _send_event(
                                    openai_responses_types.ResponseReasoningSummaryPartDoneEvent(
                                        type="response.reasoning_summary_part.done",
                                        item_id=current_item_id,
                                        sequence_number=-1,
                                        output_index=current_output_index,
                                        summary_index=0,
                                        part=openai_responses_types.response_reasoning_summary_part_done_event.Part(
                                            type="summary_text",
                                            text=simple_ctx_reasoning_text,
                                        ),
                                    )
                                )
                                yield _send_event(
                                    openai_responses_types.ResponseOutputItemDoneEvent(
                                        type="response.output_item.done",
                                        sequence_number=-1,
                                        output_index=current_output_index,
                                        item=openai_responses_types.ResponseReasoningItem(
                                            id=current_item_id,
                                            type="reasoning",
                                            status="completed",
                                            summary=[
                                                openai_responses_types.response_reasoning_item.Summary(
                                                    type="summary_text",
                                                    text=simple_ctx_reasoning_text,
                                                ),
                                            ],
                                        ),
                                    )
                                )
                                simple_ctx_sent_reasoning_done = True

                        if new_chunk:
                            # Parse for function calls if parser is available
                            if streaming_fc_parser:
                                if isinstance(streaming_fc_parser, JsonArrayParser):
                                    result = (
                                        streaming_fc_parser.parse_streaming_increment(
                                            new_chunk, function_tools
                                        )
                                    )
                                    normal_text, tool_calls = (
                                        result.normal_text,
                                        result.calls,
                                    )
                                else:
                                    normal_text, tool_calls = (
                                        streaming_fc_parser.parse_stream_chunk(
                                            new_chunk
                                        )
                                    )

                                # Emit text delta for normal text
                                if normal_text:
                                    if not simple_ctx_sent_message_added:
                                        simple_ctx_sent_message_added = True
                                        current_output_index += 1
                                        current_item_id = f"item_{random_uuid()}"
                                        yield _send_event(
                                            openai_responses_types.ResponseOutputItemAddedEvent(
                                                type="response.output_item.added",
                                                sequence_number=-1,
                                                output_index=current_output_index,
                                                item=openai_responses_types.ResponseOutputMessage(
                                                    id=current_item_id,
                                                    type="message",
                                                    role="assistant",
                                                    content=[],
                                                    status="in_progress",
                                                ),
                                            )
                                        )
                                        yield _send_event(
                                            openai_responses_types.ResponseContentPartAddedEvent(
                                                type="response.content_part.added",
                                                sequence_number=-1,
                                                output_index=current_output_index,
                                                item_id=current_item_id,
                                                content_index=current_content_index,
                                                part=openai_responses_types.ResponseOutputText(
                                                    type="output_text",
                                                    text="",
                                                    annotations=[],
                                                    logprobs=None,
                                                ),
                                            )
                                        )
                                    simple_ctx_accumulated_text += normal_text
                                    yield _send_event(
                                        openai_responses_types.ResponseTextDeltaEvent(
                                            type="response.output_text.delta",
                                            sequence_number=-1,
                                            content_index=current_content_index,
                                            output_index=current_output_index,
                                            item_id=current_item_id,
                                            delta=normal_text,
                                            logprobs=(
                                                streaming_logprobs
                                                if streaming_logprobs is not None
                                                else []
                                            ),
                                        )
                                    )
                                else:
                                    if simple_ctx_sent_message_added:
                                        simple_ctx_sent_message_added = False
                                        content = (
                                            openai_responses_types.ResponseOutputText(
                                                type="output_text",
                                                text=simple_ctx_accumulated_text,
                                                annotations=[],
                                            )
                                        )
                                        yield _send_event(
                                            openai_responses_types.ResponseContentPartDoneEvent(
                                                type="response.content_part.done",
                                                sequence_number=-1,
                                                item_id=current_item_id,
                                                output_index=current_output_index,
                                                content_index=current_content_index,
                                                part=content,
                                            )
                                        )
                                        yield _send_event(
                                            openai_responses_types.ResponseOutputItemDoneEvent(
                                                type="response.output_item.done",
                                                sequence_number=-1,
                                                output_index=current_output_index,
                                                item=openai_responses_types.ResponseOutputMessage(
                                                    id=current_item_id,
                                                    type="message",
                                                    role="assistant",
                                                    content=[content],
                                                    status="completed",
                                                ),
                                            )
                                        )

                                # Emit function call events for detected tool calls
                                for call_info in tool_calls:
                                    tool_index = call_info.tool_index

                                    args_str = call_info.parameters or ""
                                    if isinstance(args_str, dict):
                                        args_str = json.dumps(args_str)

                                    if call_info.name:
                                        fc_call_id = self._process_tool_call_id(
                                            call_info, history_tool_calls_cnt
                                        )
                                        function_name = call_info.name
                                    else:
                                        fc_call_id = ""
                                        function_name = ""

                                    if tool_index not in tool_call_states:
                                        fc_item_id = f"fc_{random_uuid()}"
                                        tool_call_states[tool_index] = {
                                            "item_id": fc_item_id,
                                            "call_id": fc_call_id,
                                            "name": function_name,
                                            "arguments": args_str,
                                            "emitted_done": False,
                                        }
                                        current_output_index += 1
                                        yield _send_event(
                                            openai_responses_types.ResponseOutputItemAddedEvent(
                                                type="response.output_item.added",
                                                sequence_number=-1,
                                                output_index=current_output_index,
                                                item=ResponseFunctionToolCall(
                                                    id=fc_item_id,
                                                    type="function_call",
                                                    call_id=fc_call_id,
                                                    name=function_name,
                                                    arguments="",
                                                    status="in_progress",
                                                ),
                                            )
                                        )
                                    else:
                                        # update existing function calls
                                        state = tool_call_states[tool_index]
                                        if function_name and not state["name"]:
                                            state["name"] = function_name
                                        if args_str:
                                            state["arguments"] += args_str

                                    current_state = tool_call_states[tool_index]
                                    # Emit arguments delta
                                    if args_str:
                                        yield _send_event(
                                            openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                                                type="response.function_call_arguments.delta",
                                                sequence_number=-1,
                                                item_id=current_state["item_id"],
                                                output_index=current_output_index,
                                                delta=args_str,
                                            )
                                        )

                                    if (
                                        not current_state["emitted_done"]
                                        and current_state["arguments"]
                                        and _is_complete_json(
                                            current_state["arguments"]
                                        )
                                    ):
                                        # Emit function call done
                                        yield _send_event(
                                            openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
                                                type="response.function_call_arguments.done",
                                                sequence_number=-1,
                                                item_id=current_state["item_id"],
                                                output_index=current_output_index,
                                                name=current_state["name"],
                                                arguments=current_state["arguments"],
                                            )
                                        )
                                        yield _send_event(
                                            openai_responses_types.ResponseOutputItemDoneEvent(
                                                type="response.output_item.done",
                                                sequence_number=-1,
                                                output_index=current_output_index,
                                                item=ResponseFunctionToolCall(
                                                    id=current_state["item_id"],
                                                    type="function_call",
                                                    call_id=current_state["call_id"],
                                                    name=current_state["name"],
                                                    arguments=current_state[
                                                        "arguments"
                                                    ],
                                                    status="completed",
                                                ),
                                            )
                                        )
                                        current_state["emitted_done"] = True
                            else:
                                # No function call parser, just emit text deltas
                                if not simple_ctx_sent_message_added:
                                    current_output_index += 1
                                    current_item_id = f"item_{random_uuid()}"
                                    simple_ctx_sent_message_added = True
                                    yield _send_event(
                                        openai_responses_types.ResponseOutputItemAddedEvent(
                                            type="response.output_item.added",
                                            sequence_number=-1,
                                            output_index=current_output_index,
                                            item=openai_responses_types.ResponseOutputMessage(
                                                id=current_item_id,
                                                type="message",
                                                role="assistant",
                                                content=[],
                                                status="in_progress",
                                            ),
                                        )
                                    )
                                    yield _send_event(
                                        openai_responses_types.ResponseContentPartAddedEvent(
                                            type="response.content_part.added",
                                            sequence_number=-1,
                                            output_index=current_output_index,
                                            item_id=current_item_id,
                                            content_index=current_content_index,
                                            part=openai_responses_types.ResponseOutputText(
                                                type="output_text",
                                                text="",
                                                annotations=[],
                                                logprobs=None,
                                            ),
                                        )
                                    )
                                simple_ctx_accumulated_text += new_chunk
                                yield _send_event(
                                    openai_responses_types.ResponseTextDeltaEvent(
                                        type="response.output_text.delta",
                                        sequence_number=-1,
                                        content_index=current_content_index,
                                        output_index=current_output_index,
                                        item_id=current_item_id,
                                        delta=new_chunk,
                                        logprobs=(
                                            streaming_logprobs
                                            if streaming_logprobs is not None
                                            else []
                                        ),
                                    )
                                )

                continue

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            id=f"rs_{random_uuid()}",
                            type="reasoning",
                            summary=[
                                openai_responses_types.response_reasoning_item.Summary(
                                    type="summary_text",
                                    text=previous_item.content[0].text,
                                ),
                            ],
                            content=[],
                            status="completed",
                        )
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                item_id=current_item_id,
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            )
                        )
                    elif previous_item.channel == "final":
                        text_content = openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            )
                        )

            if ctx.parser.last_content_delta:
                if (
                    ctx.parser.current_channel == "final"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        )
                    )
                elif (
                    ctx.parser.current_channel == "analysis"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        openai_responses_types.ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        )
                    )

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (
                    self.supports_browsing
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("browser.")
                ):
                    function_name = previous_item.recipient[len("browser.") :]
                    action = None
                    parsed_args = orjson.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = openai_responses_types.response_function_web_search.ActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = openai_responses_types.response_function_web_search.ActionOpenPage(
                            type="open_page",
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    elif function_name == "find":
                        action = openai_responses_types.response_function_web_search.ActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.response_function_web_search.ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )

                    # enqueue
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        )
                    )

                if (
                    self.supports_code_interpreter
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("python")
                ):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        )
                    )

        async def empty_async_generator():
            if False:
                yield

        if isinstance(context, SimpleContext) and simple_ctx_sent_message_added:
            content = openai_responses_types.ResponseOutputText(
                type="output_text",
                text=simple_ctx_accumulated_text,
                annotations=[],
            )
            yield _send_event(
                openai_responses_types.ResponseContentPartDoneEvent(
                    type="response.content_part.done",
                    sequence_number=-1,
                    item_id=current_item_id,
                    output_index=current_output_index,
                    content_index=current_content_index,
                    part=content,
                )
            )
            yield _send_event(
                openai_responses_types.ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=-1,
                    output_index=current_output_index,
                    item=openai_responses_types.ResponseOutputMessage(
                        id=current_item_id,
                        type="message",
                        role="assistant",
                        content=[content],
                        status="completed",
                    ),
                )
            )

        final_response = await self.responses_full_generator(
            request,
            sampling_params,
            empty_async_generator(),
            context,
            model_name,
            tokenizer,
            request_metadata,
            created_time=created_time,
        )

        # Align the call_id of the tool call item.
        if tool_call_states:
            tool_index = 0
            if isinstance(final_response, ResponsesResponse):
                for output_item in final_response.output:
                    if isinstance(output_item, ResponseFunctionToolCall):
                        if tool_index in tool_call_states:
                            current_state = tool_call_states[tool_index]
                            output_item.call_id = current_state["call_id"]
                            tool_index += 1

        # Convert final_response to the format expected by ResponseCompletedEvent
        response_dict = final_response.model_dump()

        # Convert UsageInfo to ResponseUsage format
        # if response_dict.get("usage"):
        #     usage_info = response_dict["usage"]
        #     response_dict["usage"] = {
        #         "input_tokens": usage_info.get("prompt_tokens", 0),
        #         "input_tokens_details": {
        #             "cached_tokens": usage_info.get("cached_tokens", 0)
        #         },
        #         "output_tokens": usage_info.get("completion_tokens", 0),
        #         "output_tokens_details": {
        #             "reasoning_tokens": usage_info.get("reasoning_tokens", 0)
        #         },
        #         "total_tokens": usage_info.get("total_tokens", 0),
        #     }

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        priority: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        orig_priority = priority or 0

        while True:
            # Generate using SGLang's tokenizer manager
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )

            async for res in generator:
                context.append_output(res)
                # NOTE(woosuk): The stop condition is handled by the engine.
                yield context

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_output(tool_output)

            # Prepare for the next generation turn
            # Render the updated conversation for the next completion
            prompt_token_ids = context.render_for_completion()

            # Update the adapted request with new prompt
            adapted_request = GenerateReqInput(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                stream=adapted_request.stream,
                rid=request_id,
                extra_key=adapted_request.extra_key,
                return_logprob=adapted_request.return_logprob,
                logprob_start_len=adapted_request.logprob_start_len,
                top_logprobs_num=adapted_request.top_logprobs_num,
                return_text_in_logprobs=adapted_request.return_text_in_logprobs,
                return_hidden_states=adapted_request.return_hidden_states,
                background=adapted_request.background,
                require_reasoning=adapted_request.require_reasoning,
                bootstrap_host=adapted_request.bootstrap_host,
                bootstrap_port=adapted_request.bootstrap_port,
                bootstrap_room=adapted_request.bootstrap_room,
                routed_dp_rank=adapted_request.routed_dp_rank,
                disagg_prefill_dp_rank=adapted_request.disagg_prefill_dp_rank,
                image_data=adapted_request.image_data,
                video_data=adapted_request.video_data,
                audio_data=adapted_request.audio_data,
                modalities=adapted_request.modalities,
                priority=adapted_request.priority,
                routing_key=adapted_request.routing_key,
                custom_labels=adapted_request.custom_labels,
            )

            # Update sampling params with reduced max_tokens
            if hasattr(sampling_params, "max_new_tokens") or isinstance(
                sampling_params, dict
            ):
                context_len = getattr(
                    self.tokenizer_manager.model_config, "context_len", 4096
                )
                num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                remaining_tokens = (
                    context_len - len(prompt_token_ids) - num_reserved_tokens
                )

                if isinstance(sampling_params, dict):
                    sampling_params["max_new_tokens"] = max(remaining_tokens, 1)
                else:
                    sampling_params.max_new_tokens = max(remaining_tokens, 1)

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1

    def _get_reasoning_from_request(self, request: ResponsesRequest) -> bool:
        """Judge whether the request needs reasoning for hybrid reasoning models
        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False
        if self.reasoning_parser in ["deepseek-v3"]:
            # Models that require explicit enable thinking (thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("thinking") is True
            )
        if self.reasoning_parser in ["kimi_k2"]:
            # Models that thinking by default, and can be disabled by setting thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("thinking") is not False
            )
        if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
            # Models that thinking by default, and can be disabled by setting enable_thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("enable_thinking") is not False
            )
        if self.reasoning_parser in ["mistral"]:
            # Mistral models only reason when reasoning_effort is explicitly
            # set to a value other than None/"none" (typically "high").
            return (
                request.reasoning is not None and request.reasoning.effort is not None
            )
        return True  # default

    def _get_history_tool_calls_cnt(self, request: ResponsesRequest) -> int:
        """Counts the number of tool calls in the request's input history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2.

        For the Responses API, tool calls can appear in:
        1. input items with type "function_call"
        2. assistant-role messages with tool_calls in input
        """
        if isinstance(request.input, str):
            return 0

        idx = 0
        for raw_item in request.input:
            # Convert Pydantic models to dicts for uniform handling
            item = raw_item
            if hasattr(raw_item, "model_dump"):
                item = raw_item.model_dump()

            item_type = item.get("type", "message")
            if item_type == "function_call":
                idx += 1

        return idx

    def _parse_tool_calls(
        self,
        content: str,
        tools: list[Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        request: Optional[ResponsesRequest] = None,
    ) -> tuple[str, Optional[list[ResponseFunctionToolCall]]]:
        """Process tool calls in the response"""

        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)
        history_tool_calls_cnt = (
            self._get_history_tool_calls_cnt(request) if request else 0
        )

        if self.tool_call_parser:
            parser = FunctionCallParser(tools, self.tool_call_parser)
            should_try_parser = (
                not is_required or parser.detector.supports_structural_tag()
            )
            if should_try_parser:
                remaining_text, tool_calls = parse_tool_calls_from_content(
                    content=content,
                    tools=tools,
                    tool_call_parser=self.tool_call_parser,
                    generate_tool_call_id=self._process_tool_call_id,
                    history_tool_calls_cnt=history_tool_calls_cnt,
                )
                return remaining_text, tool_calls

        if is_required:
            try:
                tool_call_data = orjson.loads(content)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    call_info = ToolCallItem(
                        tool_index=i,
                        name=tool["name"],
                        parameters=json.dumps(tool["parameters"], ensure_ascii=False),
                    )
                    call_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    # Create a ToolCallItem from the JSON data
                    function_tool_call = ResponseFunctionToolCall(
                        id=f"fc_{random_uuid()[:32]}",
                        type="function_call",
                        call_id=call_id,
                        name=tool["name"],
                        arguments=json.dumps(tool["parameters"], ensure_ascii=False),
                        status="completed",
                    )
                    tool_calls.append(function_tool_call)
                return "", tool_calls
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                return content, None

        return content, None

    def _should_return_logprobs(self, request: ResponsesRequest) -> bool:
        """Check if logprobs should be returned based on include parameter."""
        return (
            request.include is not None
            and "message.output_text.logprobs" in request.include
        )

    def _build_response_logprobs(
        self, ret_item: dict[str, Any]
    ) -> Optional[list[dict]]:
        """Convert logprobs for ResponseOutputText format."""

        meta_info = ret_item.get("meta_info", {})
        if "output_token_logprobs" not in meta_info:
            return None

        logprobs = to_openai_style_logprobs(
            output_token_logprobs=meta_info["output_token_logprobs"],
            output_top_logprobs=meta_info.get("output_top_logprobs", None),
        )

        # 转换为 ResponseOutputText.logprobs 所需格式
        token_logprobs = self._parse_logprobs_tokens(logprobs, use_token_index=True)
        return token_logprobs

    def _process_streaming_logprobs_for_responses(
        self,
        output: dict[str, Any],
        n_prev_token: int,
    ) -> Optional[list[dict]]:
        """Process logprobs for streaming response in Responses API format.

        Args:
            output: The output dict containing meta_info with logprobs data
            n_prev_token: Number of previously processed tokens

        Returns:
            List of logprob dicts for ResponseTextDeltaEvent, or None if not available
        """
        meta_info = output.get("meta_info", {})

        if "output_token_logprobs" not in meta_info:
            return None

        output_token_logprobs = meta_info["output_token_logprobs"]
        output_top_logprobs = meta_info.get("output_top_logprobs", [])

        # 获取已处理的总 token 数
        total_output_logprobs = meta_info["output_token_logprobs_length"]

        # 只处理新增的 tokens
        if n_prev_token >= total_output_logprobs:
            return None

        # 切片获取新增的 logprobs
        if not self.tokenizer_manager.server_args.incremental_streaming_output:
            output_token_logprobs = output_token_logprobs[
                n_prev_token:total_output_logprobs
            ]
            output_top_logprobs = (
                output_top_logprobs[n_prev_token:total_output_logprobs]
                if output_top_logprobs
                else []
            )

        logprobs = to_openai_style_logprobs(
            output_token_logprobs=output_token_logprobs,
            output_top_logprobs=output_top_logprobs,
        )

        # 转换为 ResponseTextDeltaEvent.logprobs 所需格式
        # 格式: List[Dict] 每个元素包含 token, logprob, bytes, top_logprobs
        token_logprobs = self._parse_logprobs_tokens(logprobs, use_token_index=True)
        return token_logprobs

    def _parse_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> Optional[list[dict]]:
        """Common helper to process logprobs tokens for both streaming and non-streaming"""
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - Non-streaming (use_token_index=True): uses token_idx for full data
                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        {
                            "token": top_token,
                            "bytes": top_token_bytes,
                            "logprob": top_logprob,
                        }
                    )
            token_logprobs.append(
                {
                    "token": token,
                    "bytes": token_bytes,
                    "logprob": logprob,
                    "top_logprobs": top_logprobs,
                }
            )

        return token_logprobs
