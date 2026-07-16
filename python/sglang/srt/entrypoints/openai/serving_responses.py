# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
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
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as ResponseReasoningSummaryAddedPart,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as ResponseReasoningSummaryDonePart,
)
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
    MessageProcessingResult,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import random_uuid

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)


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

        # Parent OpenAIServingChat.__init__ already populated default_sampling_params.
        if not isinstance(self.default_sampling_params, dict):
            self.default_sampling_params = {}

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
            str, Union[list[ChatCompletionMessageParam], list[OpenAIMessage]]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _has_response_tool(request: ResponsesRequest, *tool_types: str) -> bool:
        return any(tool.type in tool_types for tool in (request.tools or []))

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

        # ``tool_choice="required"`` only works with ``function`` tools.
        if request.tool_choice == "required" and not any(
            tool.type == "function" for tool in (request.tools or [])
        ):
            return self.create_error_response(
                'tool_choice="required" requires at least one tool with '
                'type="function"; other built-in tool types cannot be forced.'
            )

        if (
            self.use_harmony
            and self._has_response_tool(request, "web_search", "web_search_preview")
            and not self.supports_browsing
        ):
            return self.create_error_response(
                "web_search requires a browser backend. Set EXA_API_KEY on the "
                "SGLang server to enable native Exa-backed web search, or "
                "configure a browser MCP tool server. Create an Exa API key at "
                "https://dashboard.exa.ai/api-keys."
            )

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
            processed_messages: Optional[MessageProcessingResult] = None

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                (
                    messages,
                    request_prompts,
                    engine_prompts,
                    processed_messages,
                ) = await self._make_request(request, prev_response, tokenizer)

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
                tool.type in ("web_search", "web_search_preview", "code_interpreter")
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
                        default_max_tokens,
                        self.default_sampling_params,
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

                    # Create GenerateReqInput for SGLang
                    if isinstance(engine_prompt, str):
                        prompt_kwargs = {"text": engine_prompt}
                    else:
                        prompt_kwargs = {"input_ids": engine_prompt}

                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        image_data=(
                            processed_messages.image_data
                            if processed_messages
                            else None
                        ),
                        video_data=(
                            processed_messages.video_data
                            if processed_messages
                            else None
                        ),
                        audio_data=(
                            processed_messages.audio_data
                            if processed_messages
                            else None
                        ),
                        modalities=(
                            processed_messages.modalities
                            if processed_messages
                            else None
                        ),
                        sampling_params=sampling_params,
                        stream=request.stream,
                        rid=request.request_id,
                        session_id=request.session_id,
                        extra_key=self._compute_extra_key(request),
                        background=request.background,
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
                if self.use_harmony:
                    return self.responses_stream_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                    )
                return self.responses_stream_generator_non_harmony(
                    request,
                    sampling_params,
                    result_generator,
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
        messages = self._construct_input_messages(request, prev_response)

        chat_tools = self._response_tools_to_chat_tools(request)
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            stream=request.stream,
            tools=chat_tools or None,
            tool_choice=request.tool_choice if chat_tools else "none",
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            stop=request.stop,
        )

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._process_messages(chat_request, is_multimodal)

        if is_multimodal:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]

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
        return messages, [prompt_token_ids], [engine_prompt]

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
            # num_reasoning_tokens isn't wired through HarmonyContext yet; stays 0.
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            output = self._make_response_output_items(
                request, final_res["text"], tokenizer
            )

            # Calculate usage from actual output
            num_reasoning_tokens = 0
            meta_info = None
            if isinstance(final_res, dict) and isinstance(
                final_res.get("meta_info"), dict
            ):
                meta_info = final_res["meta_info"]
            elif hasattr(final_res, "meta_info"):
                meta_info = final_res.meta_info

            if meta_info is not None:
                num_prompt_tokens = meta_info.get("prompt_tokens", 0)
                num_generated_tokens = meta_info.get("completion_tokens", 0)
                num_cached_tokens = meta_info.get("cached_tokens", 0)
                num_reasoning_tokens = meta_info.get("reasoning_tokens", 0)
            elif isinstance(final_res, dict) and (
                final_res.get("prompt_token_ids") is not None
                or final_res.get("output_ids") is not None
            ):
                prompt_token_ids = final_res.get("prompt_token_ids") or []
                output_token_ids = final_res.get("output_ids") or []
                num_prompt_tokens = len(prompt_token_ids)
                num_generated_tokens = len(output_token_ids)
                num_cached_tokens = final_res.get("num_cached_tokens", 0)
            elif hasattr(final_res, "prompt_token_ids") and hasattr(
                final_res, "outputs"
            ):
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res.prompt_token_ids) if final_res.prompt_token_ids else 0
                )
                num_generated_tokens = (
                    len(final_res.outputs[0].token_ids)
                    if final_res.outputs and final_res.outputs[0].token_ids
                    else 0
                )
                num_cached_tokens = getattr(final_res, "num_cached_tokens", 0)
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0
                num_reasoning_tokens = 0

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=num_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
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

    @staticmethod
    def _wants_reasoning_summary(request: ResponsesRequest) -> bool:
        return request.reasoning is not None and request.reasoning.summary is not None

    def _is_thinking_enabled_for_request(self, request: ResponsesRequest) -> bool:
        """Whether to start the reasoning detector in thinking mode."""
        if not self.reasoning_parser:
            return False
        effort = request.reasoning.effort if request.reasoning is not None else None
        if self.reasoning_parser == "hunyuan":
            return effort not in (None, "none", "no_think")
        if self.template_manager.force_reasoning:
            return True
        config = self.template_manager.reasoning_config
        if config is None:
            # Parser-only models (DeepSeek-R1, …) carry the thinking default in
            # the detector itself.
            detector = getattr(self, "_reasoning_detector", None)
            mode = getattr(detector, "reasoning_default", None) if detector else None
            if mode is None or mode == "always":
                return mode == "always"
            if mode == "mistral":
                return effort is not None and effort != "none"
            if mode in ("thinking", "enable_thinking"):
                return effort != "none"
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                return False
            return False
        if config.special_case == "always":
            return True
        if config.special_case == "mistral":
            return effort is not None and effort != "none"
        if config.toggle_param is None or config.default_enabled is None:
            return False
        if effort == "none":
            return False
        return bool(config.default_enabled)

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
    ):
        if self.reasoning_parser:
            # Templates that prefill ``<think>`` only emit the close tag, so
            # start the detector in thinking mode.
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            # Mirror the single parsed blob into ``summary`` when the caller opts
            # in via ``reasoning.summary``; full trace stays in ``content``.
            wants_summary = self._wants_reasoning_summary(request)
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=(
                    [
                        ResponseReasoningSummary(
                            type="summary_text", text=reasoning_content
                        )
                    ]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(
                        type="reasoning_text", text=reasoning_content
                    ),
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        chat_tools = self._response_tools_to_chat_tools(request)
        is_required = request.tool_choice == "required"
        tool_call_items: list[ResponseFunctionToolCall] = []
        parsed_via_native = False
        if (
            content
            and chat_tools
            and self.tool_call_parser
            and request.tool_choice != "none"
        ):
            parser = FunctionCallParser(
                chat_tools,
                self.tool_call_parser,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            should_try_native = (
                not is_required or parser.detector.supports_structural_tag()
            )
            if should_try_native and parser.has_tool_call(content):
                try:
                    content, call_info_list = parser.parse_non_stream(content)
                    for call_info in call_info_list:
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=call_info.parameters or "",
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=call_info.name,
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    parsed_via_native = bool(call_info_list)
                except Exception as e:
                    logger.error("Tool call parsing error: %s", e)

        if content and chat_tools and is_required and not parsed_via_native:
            try:
                tool_call_data = orjson.loads(content)
                if isinstance(tool_call_data, dict):
                    tool_call_data = [tool_call_data]
                if isinstance(tool_call_data, list):
                    for tool in tool_call_data:
                        if not isinstance(tool, dict) or "name" not in tool:
                            continue
                        arguments = json.dumps(
                            tool.get("parameters", {}), ensure_ascii=False
                        )
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=arguments,
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=tool["name"],
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    content = ""
            except Exception as e:
                logger.error("Required tool JSON parse error: %s", e)

        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )
            output_items.append(message)
        output_items.extend(tool_call_items)
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

    @staticmethod
    def _response_tools_to_chat_tools(request: ResponsesRequest) -> list[Tool]:
        # Only ``function`` tools flow to chat; built-ins go through harmony.
        chat_tools = []
        for tool in request.tools:
            if tool.type != "function":
                continue
            chat_tools.append(
                Tool(
                    type="function",
                    function=Function(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                        strict=tool.strict,
                    ),
                )
            )
        return chat_tools

    @staticmethod
    def _normalize_response_content_part_for_chat(content_part: Any) -> Any:
        # Default detail=\"auto\" and lift flat min/max_dynamic_patch onto
        # image_url so the image preprocessor sees them.
        if hasattr(content_part, "model_dump"):
            content_part = content_part.model_dump(exclude_none=True)
        if not isinstance(content_part, dict):
            return content_part

        part_type = content_part.get("type")
        if part_type in ("input_text", "output_text"):
            return {"type": "text", "text": content_part.get("text", "")}

        if part_type == "input_image":
            image_url = content_part.get("image_url")
            if isinstance(image_url, dict):
                image_url_obj = image_url.copy()
            else:
                image_url_obj = {"url": image_url}
            if not image_url_obj.get("detail"):
                image_url_obj["detail"] = content_part.get("detail") or "auto"
            for key in ("min_dynamic_patch", "max_dynamic_patch"):
                if key in content_part and key not in image_url_obj:
                    image_url_obj[key] = content_part[key]
            return {"type": "image_url", "image_url": image_url_obj}

        if part_type == "text":
            return content_part

        if part_type == "image_url":
            image_url = content_part.get("image_url")
            if isinstance(image_url, str):
                image_url = {
                    "url": image_url,
                    "detail": content_part.get("detail", "auto"),
                }
            elif isinstance(image_url, dict):
                image_url = image_url.copy()
                if not image_url.get("detail"):
                    image_url["detail"] = content_part.get("detail") or "auto"
            return {**content_part, "image_url": image_url}

        return content_part

    @classmethod
    def _normalize_response_message_for_chat(cls, message: Any) -> Any:
        """Convert one Responses-API input item to a chat-completions message."""
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            return message

        # Most chat templates only recognize system/user/assistant/tool;
        # collapse ``developer`` to ``system`` at the boundary.
        if message.get("role") == "developer":
            message = {**message, "role": "system"}

        msg_type = message.get("type")
        if msg_type == "function_call":
            # Coerce ``arguments`` to a valid JSON-object string so the chat
            # template's unconditional ``orjson.loads`` survives truncated or
            # dict-shaped echoes.
            raw = message.get("arguments")
            if isinstance(raw, str):
                try:
                    parsed = orjson.loads(raw) if raw else None
                except orjson.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    raw = "{}"
            elif isinstance(raw, dict):
                raw = orjson.dumps(raw).decode("utf-8")
            else:
                raw = "{}"
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": raw,
                        },
                    }
                ],
            }
        if msg_type == "function_call_output":
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id"),
                "content": message.get("output", ""),
            }
        # Reasoning items render as {role: assistant, reasoning_content};
        # empty ones drop instead of injecting an empty assistant block.
        if msg_type == "reasoning":
            # Prefer ``summary``; fall back to ``content`` only when summary
            # is empty, since clients often populate both with the same text.
            def _collect(parts):
                out: list[str] = []
                for entry in parts or []:
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if text:
                            out.append(text)
                return out

            text_parts = _collect(message.get("summary"))
            if not text_parts:
                text_parts = _collect(message.get("content"))
            if not text_parts:
                return None
            return {
                "role": "assistant",
                "reasoning_content": "\n".join(text_parts),
            }
        if msg_type not in (None, "message"):
            raise ValueError(f"Unsupported Responses API input item type: {msg_type!r}")

        content = message.get("content")
        if not isinstance(content, list):
            return {
                k: v
                for k, v in message.items()
                if v is not None and k not in ("id", "status", "type")
            }

        return {
            k: v
            for k, v in {
                **message,
                "content": [
                    cls._normalize_response_content_part_for_chat(part)
                    for part in content
                ],
            }.items()
            if v is not None and k not in ("id", "status", "type")
        }

    @staticmethod
    def _output_message_text(output_item: Any) -> Optional[str]:
        """Return assistant text from a ``message`` output item (joining
        ``output_text`` parts with newlines), or None for non-message items."""
        if isinstance(output_item, ResponseReasoningItem):
            return None
        if hasattr(output_item, "model_dump"):
            output_item = output_item.model_dump(exclude_none=True)
        if not isinstance(output_item, dict):
            return None
        if output_item.get("type") != "message":
            return None

        text_parts = []
        for content in output_item.get("content") or []:
            if isinstance(content, ResponseOutputText):
                text_parts.append(content.text)
                continue
            if hasattr(content, "model_dump"):
                content = content.model_dump(exclude_none=True)
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if text is not None:
                    text_parts.append(text)

        return "\n".join(text_parts) if text_parts else None

    @staticmethod
    def _merge_consecutive_assistant_messages(
        messages: list,
    ) -> list:
        """Collapse runs of consecutive ``assistant`` dicts into one entry,
        joining ``content`` and concatenating ``tool_calls`` and
        ``reasoning_content`` so a logical turn renders as a single block."""
        merged: list = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and merged
                and isinstance(merged[-1], dict)
                and merged[-1].get("role") == "assistant"
            ):
                prev = merged[-1] = dict(merged[-1])
                # Lift mixed str/list content to list parts so non-text parts
                # (e.g. image_url) survive when the two sides differ in shape.
                new_content = msg.get("content")
                if new_content is not None and new_content != "":
                    prev_content = prev.get("content")
                    if prev_content is None or prev_content == "":
                        prev["content"] = new_content
                    elif isinstance(prev_content, str) and isinstance(new_content, str):
                        sep = "\n\n" if prev_content and new_content else ""
                        prev["content"] = prev_content + sep + new_content
                    else:

                        def _as_parts(c):
                            if isinstance(c, list):
                                return list(c)
                            if isinstance(c, str) and c:
                                return [{"type": "text", "text": c}]
                            return []

                        prev["content"] = _as_parts(prev_content) + _as_parts(
                            new_content
                        )
                new_calls = msg.get("tool_calls")
                if new_calls:
                    prev_calls = prev.get("tool_calls") or []
                    prev["tool_calls"] = prev_calls + list(new_calls)
                new_reasoning = msg.get("reasoning_content")
                if new_reasoning:
                    prev_reasoning = prev.get("reasoning_content")
                    prev["reasoning_content"] = (
                        f"{prev_reasoning}\n{new_reasoning}"
                        if prev_reasoning
                        else new_reasoning
                    )
                continue
            merged.append(msg)
        return merged

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

            for output_item in prev_response.output:
                assistant_text = self._output_message_text(output_item)
                if assistant_text is None:
                    continue
                messages.append({"role": "assistant", "content": assistant_text})

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for input_item in request.input:
                normalized = self._normalize_response_message_for_chat(input_item)
                if normalized is not None:
                    messages.append(normalized)  # type: ignore

        # One Responses-API assistant turn maps to multiple input items
        # (message + function_call(s)); collapse them into one chat message
        # so chat templates render a single assistant block per turn.
        messages = self._merge_consecutive_assistant_messages(messages)

        # Most chat templates expect a single leading ``system`` message;
        # coalesce any ``instructions`` + interleaved ``developer`` entries.
        system_chunks: list[str] = []
        other_msgs: list = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                system_chunks.append(text)
            else:
                other_msgs.append(m)
        if system_chunks:
            return [
                {"role": "system", "content": "\n\n".join(system_chunks)}
            ] + other_msgs
        return other_msgs

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[OpenAIMessage]:
        messages: list[OpenAIMessage] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                any(t in tool_types for t in ("web_search", "web_search_preview"))
                and self.tool_server is not None
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
                prev_outputs = list(prev_response.output)
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

        async for ctx in result_generator:

            # Only process context objects that implement the `is_expecting_start()` method,
            # which indicates they support per-turn streaming (e.g., StreamingHarmonyContext).
            # Contexts without this method are skipped, as they do not represent a new turn
            # or are not compatible with per-turn handling in the /v1/responses endpoint.
            if not hasattr(ctx, "is_expecting_start"):
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
                            summary=[],
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text,
                                    type="reasoning_text",
                                ),
                            ],
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
            for _ in ():
                yield

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
        # Convert final_response to the format expected by ResponseCompletedEvent
        response_dict = final_response.model_dump()
        # OpenAI SDK's Tool union may not know extended types; drop echo.
        response_dict["tools"] = []

        # Convert UsageInfo to ResponseUsage format
        if response_dict.get("usage"):
            usage_info = response_dict["usage"]
            response_dict["usage"] = {
                "input_tokens": usage_info.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": usage_info.get("cached_tokens", 0)
                },
                "output_tokens": usage_info.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": usage_info.get("reasoning_tokens", 0)
                },
                "total_tokens": usage_info.get("total_tokens", 0),
            }

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def responses_stream_generator_non_harmony(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a /v1/responses response as typed OpenAI SSE events for
        non-harmony models. Each engine chunk is run through the reasoning
        and function-call parsers; leftover text becomes
        ``response.output_text.delta``.
        """

        created_time = created_time or int(time.time())
        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n"
            )

        # The streaming Response* event models echo ``tools`` through a
        # narrower OpenAI SDK Tool union; strip it to avoid pydantic
        # validation failures on extended tool types.
        def _sanitize_response_dict(d: dict) -> dict:
            d["tools"] = []
            return d

        initial_response = _sanitize_response_dict(
            ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="in_progress",
                usage=None,
            ).model_dump()
        )
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

        chat_tools = self._response_tools_to_chat_tools(request)
        is_required = request.tool_choice == "required"
        tool_parser: Optional[Union[FunctionCallParser, JsonArrayParser]] = None
        if chat_tools and request.tool_choice != "none":
            native_supports_structural_tag = False
            if self.tool_call_parser:
                probe = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                native_supports_structural_tag = (
                    probe.detector.supports_structural_tag()
                )
            if is_required and not native_supports_structural_tag:
                tool_parser = JsonArrayParser()
            elif self.tool_call_parser:
                tool_parser = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
        reasoning_parser_obj: Optional[ReasoningParser] = None
        if self.reasoning_parser:
            reasoning_parser_obj = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                force_reasoning=self._is_thinking_enabled_for_request(request),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
            )

        current_output_index = -1
        reasoning_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        message_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        tool_call_states: dict[int, dict[str, Any]] = {}
        # Items closed during the stream, in wire order. Feeds the final
        # ``response.completed`` snapshot and the stored response.
        emitted_items: list = []

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        total_tokens_meta = 0
        reasoning_tokens_meta = 0
        finish_reason: Optional[dict[str, Any]] = None
        stream_offset = 0
        incremental = self.tokenizer_manager.server_args.incremental_streaming_output

        def _open_reasoning_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"rs_{random_uuid()}"
            reasoning_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        wants_summary = self._wants_reasoning_summary(request)

        def _close_reasoning_item():
            if not reasoning_state["open"]:
                return []
            text = reasoning_state["text"]
            completed_item = ResponseReasoningItem(
                id=reasoning_state["item_id"],
                type="reasoning",
                summary=(
                    [ResponseReasoningSummary(type="summary_text", text=text)]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(type="reasoning_text", text=text),
                ],
                status="completed",
            )
            events: list = []
            if wants_summary:
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningSummaryTextDoneEvent(
                            type="response.reasoning_summary_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            text=text,
                        )
                    )
                )
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningSummaryPartDoneEvent(
                            type="response.reasoning_summary_part.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            part=ResponseReasoningSummaryDonePart(
                                type="summary_text", text=text
                            ),
                        )
                    )
                )
            else:
                events.append(
                    _send_event(
                        openai_responses_types.ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            content_index=0,
                            text=text,
                        )
                    )
                )
            events += [
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=reasoning_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            reasoning_state["open"] = False
            return events

        def _open_message_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"msg_{random_uuid()}"
            message_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        def _close_message_item():
            if not message_state["open"]:
                return []
            text = message_state["text"]
            text_content = openai_responses_types.ResponseOutputText(
                type="output_text", text=text, annotations=[], logprobs=None
            )
            completed_item = ResponseOutputMessage(
                id=message_state["item_id"],
                type="message",
                role="assistant",
                content=[text_content],
                status="completed",
            )
            events = [
                _send_event(
                    openai_responses_types.ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        content_index=0,
                        text=text,
                        logprobs=[],
                        item_id=message_state["item_id"],
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=message_state["item_id"],
                        output_index=message_state["output_index"],
                        content_index=0,
                        part=text_content,
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            message_state["open"] = False
            return events

        def _close_tool_call_state(tool_index: int):
            state = tool_call_states.get(tool_index)
            if state is None or state.get("done"):
                return []
            arguments = state["arguments"]
            completed_item = ResponseFunctionToolCall(
                arguments=arguments,
                call_id=state["call_id"],
                name=state["name"] or "",
                type="function_call",
                id=state["item_id"],
                status="completed",
            )
            events = [
                _send_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=-1,
                        item_id=state["item_id"],
                        output_index=state["output_index"],
                        arguments=arguments,
                        name=state["name"] or "",
                    )
                ),
                _send_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            state["done"] = True
            return events

        try:
            async for ctx in result_generator:
                if isinstance(ctx, dict):
                    chunk = ctx
                else:
                    chunk = getattr(ctx, "last_output", None)
                if not isinstance(chunk, dict):
                    continue
                meta = chunk.get("meta_info") or {}
                prompt_tokens = meta.get("prompt_tokens", prompt_tokens)
                completion_tokens = meta.get("completion_tokens", completion_tokens)
                cached_tokens = meta.get("cached_tokens", cached_tokens)
                total_tokens_meta = meta.get("total_tokens", total_tokens_meta)
                reasoning_tokens_meta = meta.get(
                    "reasoning_tokens", reasoning_tokens_meta
                )
                finish_reason = meta.get("finish_reason") or finish_reason

                text = chunk.get("text", "") or ""
                if incremental:
                    delta = text
                else:
                    delta = text[stream_offset:]
                    stream_offset = len(text)
                if not delta and finish_reason is None:
                    continue

                if reasoning_parser_obj is not None:
                    reasoning_chunk, delta = reasoning_parser_obj.parse_stream_chunk(
                        delta
                    )
                else:
                    reasoning_chunk = None

                if reasoning_chunk:
                    if message_state["open"]:
                        for ev in _close_message_item():
                            yield ev
                    if not reasoning_state["open"]:
                        item_id = _open_reasoning_item()
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=reasoning_state["output_index"],
                                item=ResponseReasoningItem(
                                    id=item_id,
                                    type="reasoning",
                                    summary=[],
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        # Clients that opt into ``reasoning.summary`` render
                        # off the ``reasoning_summary_text.*`` event stream,
                        # so mirror the trace into a summary part.
                        if wants_summary:
                            yield _send_event(
                                openai_responses_types.ResponseReasoningSummaryPartAddedEvent(
                                    type="response.reasoning_summary_part.added",
                                    item_id=item_id,
                                    output_index=reasoning_state["output_index"],
                                    summary_index=0,
                                    part=ResponseReasoningSummaryAddedPart(
                                        type="summary_text", text=""
                                    ),
                                    sequence_number=-1,
                                )
                            )
                    reasoning_state["text"] += reasoning_chunk
                    if wants_summary:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningSummaryTextDeltaEvent(
                                type="response.reasoning_summary_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                summary_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )
                    else:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDeltaEvent(
                                type="response.reasoning_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                content_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )

                if not delta:
                    continue

                if isinstance(tool_parser, JsonArrayParser):
                    sp = tool_parser.parse_streaming_increment(delta, chat_tools)
                    normal_text, tool_calls = sp.normal_text or "", sp.calls
                elif tool_parser is not None:
                    normal_text, tool_calls = tool_parser.parse_stream_chunk(delta)
                else:
                    normal_text, tool_calls = delta, []

                # Close any open tool-call item before opening a message so
                # ``output_item.done`` lands before the next ``added``.
                if normal_text:
                    if reasoning_state["open"]:
                        for ev in _close_reasoning_item():
                            yield ev
                    for tool_index in list(tool_call_states):
                        for ev in _close_tool_call_state(tool_index):
                            yield ev
                    if not message_state["open"]:
                        item_id = _open_message_item()
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=message_state["output_index"],
                                item=ResponseOutputMessage(
                                    id=item_id,
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
                                output_index=message_state["output_index"],
                                item_id=message_state["item_id"],
                                content_index=0,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    message_state["text"] += normal_text
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=0,
                            output_index=message_state["output_index"],
                            item_id=message_state["item_id"],
                            delta=normal_text,
                            logprobs=[],
                        )
                    )

                if not tool_calls:
                    continue

                if reasoning_state["open"]:
                    for ev in _close_reasoning_item():
                        yield ev
                if message_state["open"]:
                    for ev in _close_message_item():
                        yield ev

                for call in tool_calls:
                    tool_index = call.tool_index
                    state = tool_call_states.get(tool_index)
                    if state is None or state.get("done"):
                        current_output_index += 1
                        item_id = f"fc_{random_uuid()[:8]}"
                        call_id = f"call_{random_uuid()[:24]}"
                        state = {
                            "item_id": item_id,
                            "call_id": call_id,
                            "output_index": current_output_index,
                            "name": call.name or "",
                            "arguments": "",
                            "added": False,
                            "done": False,
                        }
                        tool_call_states[tool_index] = state
                    if not state["added"]:
                        state["added"] = True
                        # Capture ``call.name`` before the ``added`` event so
                        # the name is set on the first emitted item.
                        if call.name and not state["name"]:
                            state["name"] = call.name
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=state["output_index"],
                                item=ResponseFunctionToolCall(
                                    arguments="",
                                    call_id=state["call_id"],
                                    name=state["name"],
                                    type="function_call",
                                    id=state["item_id"],
                                    status="in_progress",
                                ),
                            )
                        )
                    if call.parameters:
                        state["arguments"] += call.parameters
                        yield _send_event(
                            openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                                type="response.function_call_arguments.delta",
                                sequence_number=-1,
                                item_id=state["item_id"],
                                output_index=state["output_index"],
                                delta=call.parameters,
                            )
                        )
        except Exception:
            logger.exception("Error while streaming /v1/responses")
            failed = _sanitize_response_dict(
                ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="failed",
                    usage=None,
                ).model_dump()
            )
            yield _send_event(
                openai_responses_types.ResponseFailedEvent(
                    type="response.failed",
                    sequence_number=-1,
                    response=failed,
                )
            )
            return

        for ev in _close_reasoning_item():
            yield ev
        for ev in _close_message_item():
            yield ev
        for tool_index in list(tool_call_states):
            for ev in _close_tool_call_state(tool_index):
                yield ev

        final_output_items = list(emitted_items)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens_meta or (prompt_tokens + completion_tokens),
            reasoning_tokens=reasoning_tokens_meta,
        )
        if self.enable_prompt_tokens_details and cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=cached_tokens
            )
        request_metadata.final_usage_info = usage

        final_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=final_output_items,
            status="completed",
            usage=usage,
        )
        if request.store:
            async with self.response_store_lock:
                stored = self.response_store.get(final_response.id)
                if stored is None or stored.status != "cancelled":
                    self.response_store[final_response.id] = final_response

        response_dict = _sanitize_response_dict(final_response.model_dump())
        if response_dict.get("usage"):
            usage_info = response_dict["usage"]
            response_dict["usage"] = {
                "input_tokens": usage_info.get("prompt_tokens", 0),
                "input_tokens_details": {
                    "cached_tokens": cached_tokens,
                },
                "output_tokens": usage_info.get("completion_tokens", 0),
                "output_tokens_details": {
                    "reasoning_tokens": reasoning_tokens_meta,
                },
                "total_tokens": usage_info.get("total_tokens", 0),
            }

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
                session_id=adapted_request.session_id,
                extra_key=adapted_request.extra_key,
                return_logprob=adapted_request.return_logprob,
                logprob_start_len=adapted_request.logprob_start_len,
                top_logprobs_num=adapted_request.top_logprobs_num,
                return_text_in_logprobs=adapted_request.return_text_in_logprobs,
                return_hidden_states=adapted_request.return_hidden_states,
                background=adapted_request.background,
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
