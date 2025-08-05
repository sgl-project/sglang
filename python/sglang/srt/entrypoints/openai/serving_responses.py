# SPDX-License-Identifier: Apache-2.0
"""Handler for /v1/responses requests"""

import asyncio
import json
import logging
import time
import uuid
from http import HTTPStatus
from typing import Any, AsyncGenerator, AsyncIterator, Optional, Union

import jinja2
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

try:
    from openai_harmony import Message as OpenAIMessage

    from sglang.srt.entrypoints.harmony_utils import (
        get_developer_message,
        get_stop_tokens_for_assistant_actions,
        get_system_message,
        get_user_message,
        parse_output_message,
        parse_response_input,
        parse_response_output,
        render_for_completion,
    )
except ImportError:
    print("OpenAI Harmony not installed, skipping")

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
)
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    MessageProcessingResult,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponseReasoningItem,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.tool import HarmonyBrowserTool, HarmonyPythonTool
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        use_harmony: bool = True,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage

        # Get default sampling params from model config if available
        self.default_sampling_params = {}

        self.supports_browsing = False
        self.supports_code_interpreter = False
        self.use_harmony = use_harmony

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

            self.browser_tool = HarmonyBrowserTool()
            self.supports_browsing = self.browser_tool.enabled
            self.python_tool = HarmonyPythonTool()
            self.supports_code_interpreter = self.python_tool.enabled

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
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
            else:
                messages, request_prompts, engine_prompts = await self._make_request(
                    request, prev_response, tokenizer
                )

        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                # Calculate default max tokens from context length minus prompt length
                if hasattr(engine_prompt, "__len__"):
                    prompt_length = len(engine_prompt)
                elif isinstance(engine_prompt, list):
                    prompt_length = len(engine_prompt)
                else:
                    prompt_length = 0

                context_len = (
                    self.tokenizer_manager.model_config.context_len
                    if hasattr(self.tokenizer_manager.model_config, "context_len")
                    else 4096
                )
                default_max_tokens = max(
                    context_len - prompt_length, 512
                )  # Ensure minimum 512 tokens
                sampling_params = request.to_sampling_params(
                    default_max_tokens, self.default_sampling_params
                )

                if self.use_harmony:
                    context = HarmonyContext(
                        messages, self.browser_tool, self.python_tool
                    )
                else:
                    context = SimpleContext()

                # Create GenerateReqInput for SGLang
                adapted_request = GenerateReqInput(
                    input_ids=engine_prompt,
                    sampling_params=sampling_params,
                    stream=request.stream,
                    rid=request.request_id,
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
            print(f"!!!!!!! DEBUG: create_responses response.id: {response.id}")
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
            # NOTE: check proper error handling for NotImplementedError
            raise NotImplementedError("Streaming responses are not supported")

        try:
            return await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )
        except Exception as e:
            return self.create_error_response(str(e))

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        # Construct the input messages
        messages = self._construct_input_messages(request, prev_response)

        # Follow SGLang's pattern: create a ChatCompletionRequest and process messages
        try:
            # Convert ResponsesRequest to ChatCompletionRequest for processing
            chat_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                stream=request.stream,
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
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n"
            prompt_ids = tokenizer.encode(prompt_text)
            request_prompts = [prompt_ids]
            engine_prompts = [prompt_ids]

        return messages, request_prompts, engine_prompts

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
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
            output = self._make_response_output_items_with_harmony(context)
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
        else:
            final_res = context.last_output
            assert final_res is not None

            output = self._make_response_output_items(
                request, final_res["text"], tokenizer
            )

            # Calculate usage from actual output
            if hasattr(final_res, "meta_info"):
                num_prompt_tokens = final_res.meta_info.get("prompt_tokens", 0)
                num_generated_tokens = final_res.meta_info.get("completion_tokens", 0)
                num_cached_tokens = final_res.meta_info.get("cached_tokens", 0)
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

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
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

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
    ):
        # Handle reasoning parsing if enabled
        if self.reasoning_parser:
            # Use standard reasoning parser (openai maps to T4Detector internally)
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser, stream_reasoning=False
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            reasoning_item = ResponseReasoningItem(
                text=reasoning_content,
                status=None,  # NOTE: Only the last output item has status
            )
            output_items.append(reasoning_item)
        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            message = ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
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
            messages.extend(request.input)  # type: ignore
        return messages

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list["OpenAIMessage"]:
        messages: list["OpenAIMessage"] = []
        if prev_response is None:
            # New conversation
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browsing = (
                "web_search_preview" in tool_types and self.supports_browsing
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.supports_code_interpreter
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                enable_browsing=enable_browsing,
                browser_tool=self.browser_tool,
                enable_python=enable_code_interpreter,
                python_tool=self.python_tool,
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation
            # Note: For continued conversations, we reuse the existing context
            # and do not re-apply reasoning/instruction parameters to maintain consistency
            # FIXME: reasoning and instructions are better not to be ignored
            prev_msgs = self.msg_store[prev_response.id]
            messages.extend(prev_msgs)

            # Append the previous output
            for output_item in prev_response.output:
                # NOTE: We skip the reasoning output of the previous response
                if isinstance(output_item, ResponseReasoningItem):
                    continue
                messages.append(parse_response_output(output_item))

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg))
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
            response = await self.responses_full_generator(request, sampling_params, result_generator, context, model_name, tokenizer, request_metadata, created_time, *args, **kwargs)
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            print(f"!!!!!!! DEBUG: _run_background_request response_id: {response_id}")
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
                    "Cannot cancel a synchronous response.",
                    err_type="invalid_request_error",
                )

            # Update the status to "cancelled"
            response.status = "cancelled"

        # Abort the request
        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

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
            try:
                generator = self.tokenizer_manager.generate_request(
                    adapted_request, raw_request
                )

                async for res in generator:
                    # Append output to context for tool processing
                    context.append_output(res)
                    # Yield the context back for processing
                    # NOTE: check for stop conditions
                    yield context

            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise

            # Check if the model requested a tool call
            tool = context.get_tool_call()
            if tool is None:
                # No tool call requested, generation is complete
                break

            # Execute the tool and get the result
            tool_output = await tool.get_result(context)
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
                remaining_tokens = context_len - len(prompt_token_ids)

                if isinstance(sampling_params, dict):
                    sampling_params["max_new_tokens"] = max(remaining_tokens, 1)
                else:
                    sampling_params.max_new_tokens = max(remaining_tokens, 1)

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1

    async def _handle_streaming_responses(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
    ) -> StreamingResponse:
        """Handle streaming responses."""

        async def generate_stream():
            try:
                if self.use_harmony:
                    async for res in result_generator:
                        # For harmony, we process the output and yield incremental updates
                        output_items = self._make_response_output_items_with_harmony(
                            context
                        )

                        # Create a partial response
                        partial_response = {
                            "id": f"resp_{request.request_id}",
                            "object": "realtime.response.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "output": output_items,
                            "status": "in_progress",
                        }
                        yield f"data: {json.dumps(partial_response)}\n\n"
                else:
                    # For non-harmony mode, stream token by token
                    async for res in result_generator:
                        # Create streaming chunk
                        chunk = {
                            "id": f"resp_{request.request_id}",
                            "object": "realtime.response.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "delta": {"content": res.get("text", "")},
                            "status": "in_progress",
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Send final completion chunk
                final_chunk = {
                    "id": f"resp_{request.request_id}",
                    "object": "realtime.response.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "status": "completed",
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                error_chunk = {
                    "id": f"resp_{request.request_id}",
                    "object": "realtime.response.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "error": {"message": str(e), "type": "internal_error"},
                    "status": "failed",
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

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
