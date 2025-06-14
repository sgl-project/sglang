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
"""Chat completions serving logic for OpenAI API"""

import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    ToolCall,
    TopLogprob,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_engine import (
    OpenAIServingBase,
    RequestContext,
)
from sglang.srt.entrypoints.openai.utils import (
    _get_enable_thinking_from_request,
    aggregate_token_usage,
    build_base_sampling_params,
    create_error_response,
    create_stream_done,
    create_streaming_chunk_data,
    create_streaming_error_response,
    detect_template_content_format,
    process_content_for_template_format,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.utils import convert_json_schema_to_str

logger = logging.getLogger(__name__)

# Global cache for template content format detection
_cached_chat_template = None
_cached_template_format = None


class ChatCompletionHandler(OpenAIServingBase):
    """Handler for chat completion requests"""

    def _convert_to_internal_request(
        self,
        all_requests: List[ChatCompletionRequest],
        request_ids: List[str],
    ) -> tuple[
        GenerateReqInput, Union[ChatCompletionRequest, List[ChatCompletionRequest]]
    ]:
        """Convert OpenAI chat completion request to internal format"""
        input_ids = []
        prompts = []
        sampling_params_list = []
        image_data_list = []
        audio_data_list = []
        return_logprobs = []
        logprob_start_lens = []
        top_logprobs_nums = []
        modalities_list = []
        lora_paths = []

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        for request in all_requests:
            # Process messages and apply chat template
            prompt, prompt_ids, image_data, audio_data, modalities, stop = (
                self._process_messages(request, is_multimodal)
            )

            input_ids.append(prompt_ids)
            prompts.append(prompt)
            return_logprobs.append(request.logprobs)
            logprob_start_lens.append(-1)
            top_logprobs_nums.append(request.top_logprobs or 0)
            lora_paths.append(request.lora_path)

            # Build sampling parameters
            sampling_params = self._build_sampling_params(request, stop)
            sampling_params_list.append(sampling_params)

            image_data_list.append(image_data)
            audio_data_list.append(audio_data)
            modalities_list.append(modalities)

        # Handle single vs multiple requests
        if len(all_requests) == 1:
            if is_multimodal:
                prompt_kwargs = {"text": prompts[0]}
            else:
                if isinstance(input_ids[0], str):
                    prompt_kwargs = {"text": input_ids[0]}
                else:
                    prompt_kwargs = {"input_ids": input_ids[0]}

            sampling_params_list = sampling_params_list[0]
            image_data_list = image_data_list[0]
            audio_data_list = audio_data_list[0]
            return_logprobs = return_logprobs[0]
            logprob_start_lens = logprob_start_lens[0]
            top_logprobs_nums = top_logprobs_nums[0]
            modalities_list = modalities_list[0]
            lora_paths = lora_paths[0]
            request_ids = request_ids[0]
        else:
            if is_multimodal:
                prompt_kwargs = {"text": prompts}
            else:
                if isinstance(input_ids[0], str):
                    prompt_kwargs = {"text": input_ids}
                else:
                    prompt_kwargs = {"input_ids": input_ids}

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=image_data_list,
            audio_data=audio_data_list,
            sampling_params=sampling_params_list,
            return_logprob=return_logprobs,
            logprob_start_len=logprob_start_lens,
            top_logprobs_num=top_logprobs_nums,
            stream=all_requests[0].stream,
            return_text_in_logprobs=True,
            rid=request_ids,
            modalities=modalities_list,
            lora_path=lora_paths,
            bootstrap_host=all_requests[0].bootstrap_host,
            bootstrap_port=all_requests[0].bootstrap_port,
            bootstrap_room=all_requests[0].bootstrap_room,
        )

        return adapted_request, (
            all_requests if len(all_requests) > 1 else all_requests[0]
        )

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> tuple[
        str, Union[str, List[int]], Optional[Any], Optional[Any], List[str], List[str]
    ]:
        """Process chat messages and apply chat template"""
        tool_call_constraint = None
        prompt = ""
        prompt_ids = []

        if not isinstance(request.messages, str):
            # Apply chat template and its stop strings
            tools = None
            if request.tools and request.tool_choice != "none":
                request.skip_special_tokens = False
                if not isinstance(request.tool_choice, str):
                    tools = [
                        item.function.model_dump()
                        for item in request.tools
                        if item.function.name == request.tool_choice.function.name
                    ]
                else:
                    tools = [item.function.model_dump() for item in request.tools]

                tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
                parser = FunctionCallParser(request.tools, tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice
                )

            # Use chat template
            if (
                hasattr(self.tokenizer_manager, "chat_template_name")
                and self.tokenizer_manager.chat_template_name is None
            ):
                prompt, prompt_ids, image_data, audio_data, modalities, stop = (
                    self._apply_jinja_template(request, tools, is_multimodal)
                )
            else:
                prompt, image_data, audio_data, modalities, stop = (
                    self._apply_conversation_template(request)
                )
                if not is_multimodal:
                    prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)
        else:
            # Use raw prompt
            prompt_ids = request.messages
            stop = request.stop or []
            image_data = None
            audio_data = None
            modalities = []
            prompt = request.messages

        return prompt, prompt_ids, image_data, audio_data, modalities, stop

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> tuple[str, List[int], Optional[Any], Optional[Any], List[str], List[str]]:
        """Apply Jinja chat template"""
        global _cached_chat_template, _cached_template_format

        openai_compatible_messages = []
        image_data = []
        audio_data = []
        modalities = []

        # Detect template content format
        current_template = self.tokenizer_manager.tokenizer.chat_template
        if current_template != _cached_chat_template:
            _cached_chat_template = current_template
            _cached_template_format = detect_template_content_format(current_template)
            logger.info(
                f"Detected chat template content format: {_cached_template_format}"
            )

        template_content_format = _cached_template_format

        for message in request.messages:
            if message.content is None:
                message.content = ""
            msg_dict = message.model_dump()

            # Process content based on detected template format
            processed_msg = process_content_for_template_format(
                msg_dict,
                template_content_format,
                image_data,
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
            if request.continue_final_message:
                assistant_prefix = openai_compatible_messages[-1]["content"]
                openai_compatible_messages = openai_compatible_messages[:-1]

        try:
            prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                **(
                    request.chat_template_kwargs if request.chat_template_kwargs else {}
                ),
            )
        except Exception:
            # Handle different tools input format (e.g., Mistral)
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
                **(
                    request.chat_template_kwargs if request.chat_template_kwargs else {}
                ),
            )

        if assistant_prefix:
            encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
            if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
                encoded = encoded[1:]
            prompt_ids += encoded

        if is_multimodal:
            prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop or []
        return prompt, prompt_ids, image_data, audio_data, modalities, stop

    def _apply_conversation_template(
        self, request: ChatCompletionRequest
    ) -> tuple[str, Optional[Any], Optional[Any], List[str], List[str]]:
        """Apply conversation template"""
        conv = generate_chat_conv(request, self.tokenizer_manager.chat_template_name)

        # Handle continue_final_message
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            prompt = conv.get_prompt()
            # Strip trailing stop tokens
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

        image_data = conv.image_data
        audio_data = conv.audio_data
        modalities = conv.modalities
        stop = conv.stop_str or [] if not request.ignore_eos else []

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        return prompt, image_data, audio_data, modalities, stop

    def _build_sampling_params(
        self, request: ChatCompletionRequest, stop: List[str]
    ) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        # Start with common parameters
        sampling_params = build_base_sampling_params(request)

        # Override stop with processed stop sequences
        sampling_params["stop"] = stop

        # Handle response format
        if request.response_format and request.response_format.type == "json_schema":
            sampling_params["json_schema"] = convert_json_schema_to_str(
                request.response_format.json_schema.schema_
            )
        elif request.response_format and request.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif (
            request.response_format and request.response_format.type == "structural_tag"
        ):
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                request.response_format.model_dump(by_alias=True)
            )

        # Handle tool call constraints
        if hasattr(self, "_tool_call_constraint") and self._tool_call_constraint:
            constraint_type, constraint_value = self._tool_call_constraint
            if constraint_type == "structural_tag":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value.model_dump(by_alias=True)
                )
            else:
                sampling_params[constraint_type] = constraint_value

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        ctx: RequestContext,
    ) -> StreamingResponse:
        """Handle streaming chat completion request"""

        async def generate_stream_resp():
            parser_dict = {}
            reasoning_parser_dict = {}
            tool_call_first = True
            is_firsts = {}
            stream_buffers = {}
            n_prev_tokens = {}
            prompt_tokens = {}
            completion_tokens = {}
            cached_tokens = {}

            try:
                async for content in self.tokenizer_manager.generate_request(
                    adapted_request, ctx.raw_request
                ):
                    index = content.get("index", 0)
                    text = content["text"]

                    is_first = is_firsts.get(index, True)
                    stream_buffer = stream_buffers.get(index, "")
                    n_prev_token = n_prev_tokens.get(index, 0)

                    prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                    completion_tokens[index] = content["meta_info"]["completion_tokens"]
                    cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)

                    # Handle logprobs
                    choice_logprobs = None
                    if request.logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token
                        )
                        n_prev_token = len(
                            content["meta_info"]["output_token_logprobs"]
                        )

                    finish_reason = content["meta_info"]["finish_reason"]
                    finish_reason_type = (
                        finish_reason["type"] if finish_reason else None
                    )

                    # First chunk with role
                    if is_first:
                        is_first = False
                        delta = DeltaMessage(role="assistant")
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=delta,
                            finish_reason=finish_reason_type,
                            matched_stop=(
                                finish_reason["matched"]
                                if finish_reason and "matched" in finish_reason
                                else None
                            ),
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield create_streaming_chunk_data(chunk.model_dump_json())

                    # Process content delta
                    delta = text[len(stream_buffer) :]
                    new_stream_buffer = stream_buffer + delta

                    # Handle reasoning content
                    enable_thinking = _get_enable_thinking_from_request(request)
                    if (
                        self.tokenizer_manager.server_args.reasoning_parser
                        and request.separate_reasoning
                        and enable_thinking
                    ):
                        reasoning_text, delta = self._process_reasoning_stream(
                            index, delta, reasoning_parser_dict, content, request
                        )
                        if reasoning_text:
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(reasoning_content=reasoning_text),
                                finish_reason=finish_reason_type,
                            )
                            chunk = ChatCompletionStreamResponse(
                                id=content["meta_info"]["id"],
                                created=int(time.time()),
                                choices=[choice_data],
                                model=request.model,
                            )
                            yield create_streaming_chunk_data(chunk.model_dump_json())

                        if not delta or len(delta) == 0:
                            stream_buffers[index] = new_stream_buffer
                            is_firsts[index] = is_first
                            n_prev_tokens[index] = n_prev_token
                            continue

                    # Handle tool calls
                    if request.tool_choice != "none" and request.tools:
                        async for chunk in self._process_tool_call_stream(
                            index,
                            delta,
                            parser_dict,
                            content,
                            request,
                            finish_reason_type,
                        ):
                            yield chunk
                    else:
                        # Regular content
                        if delta or not (
                            request.stream_options
                            and request.stream_options.include_usage
                        ):
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(content=delta if delta else None),
                                finish_reason=(
                                    None
                                    if request.stream_options
                                    and request.stream_options.include_usage
                                    else finish_reason_type
                                ),
                                matched_stop=(
                                    finish_reason["matched"]
                                    if finish_reason and "matched" in finish_reason
                                    else None
                                ),
                                logprobs=choice_logprobs,
                            )
                            chunk = ChatCompletionStreamResponse(
                                id=content["meta_info"]["id"],
                                created=int(time.time()),
                                choices=[choice_data],
                                model=request.model,
                            )
                            yield create_streaming_chunk_data(chunk.model_dump_json())

                    stream_buffers[index] = new_stream_buffer
                    is_firsts[index] = is_first
                    n_prev_tokens[index] = n_prev_token

                # Final chunk with usage
                if request.stream_options and request.stream_options.include_usage:
                    usage = self._calculate_streaming_usage_base(
                        prompt_tokens, completion_tokens, cached_tokens, request
                    )
                else:
                    usage = None

                final_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason_type,
                        )
                    ],
                    model=request.model,
                    usage=usage,
                )
                yield create_streaming_chunk_data(final_chunk.model_dump_json())

            except Exception as e:
                error = create_streaming_error_response(str(e))
                yield create_streaming_chunk_data(error)

            yield create_stream_done()

        return StreamingResponse(
            generate_stream_resp(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        ctx: RequestContext,
    ) -> Union[ChatCompletionResponse, ErrorResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, ctx.raw_request
            ).__anext__()
        except ValueError as e:
            return create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
            cache_report=self.tokenizer_manager.server_args.enable_cache_report,
            tool_call_parser=self.tokenizer_manager.server_args.tool_call_parser,
            reasoning_parser=self.tokenizer_manager.server_args.reasoning_parser,
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
        cache_report: bool = False,
        tool_call_parser: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Build chat completion response from generation results"""
        choices = []

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            enable_thinking = _get_enable_thinking_from_request(request)
            if reasoning_parser and request.separate_reasoning and enable_thinking:
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser, stream_reasoning=False
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if request.tool_choice != "none" and request.tools:
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text, request.tools, tool_call_parser, finish_reason
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
            )
            choices.append(choice_data)

        # Calculate usage
        usage = aggregate_token_usage(ret, request.n, cache_report)

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = []
        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                for top_token, top_logprob in logprobs.top_logprobs[token_idx].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        tool_call_parser: Optional[str],
        finish_reason: Dict[str, Any],
    ) -> tuple[Optional[List[ToolCall]], str, Dict[str, Any]]:
        """Process tool calls in the response"""
        parser = FunctionCallParser(tools, tool_call_parser)
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = [
                    ToolCall(
                        id=f"call_{base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode()}",
                        function=FunctionResponse(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]
                return tool_calls, text, finish_reason
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                # Return error but don't fail the whole request
                return None, text, finish_reason

        return None, text, finish_reason

    def _process_streaming_logprobs(
        self, content: Dict[str, Any], n_prev_token: int
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:
            ],
        )

        token_logprobs = []
        for token, logprob in zip(logprobs.tokens, logprobs.token_logprobs):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                for top_token, top_logprob in logprobs.top_logprobs[0].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            reasoning_parser_dict[index] = ReasoningParser(
                self.tokenizer_manager.server_args.reasoning_parser,
                request.stream_reasoning,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        finish_reason_type: Optional[str],
    ):
        """Process tool calls in streaming response"""
        if index not in parser_dict:
            parser_dict[index] = FunctionCallParser(
                tools=request.tools,
                tool_call_parser=self.tokenizer_manager.server_args.tool_call_parser,
            )
        parser = parser_dict[index]

        normal_text, calls = parser.parse_stream_chunk(delta)

        # Yield normal text
        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=finish_reason_type,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        for call_item in calls:
            if finish_reason_type == "stop":
                # Handle remaining arguments
                latest_delta_len = 0
                if isinstance(call_item.parameters, str):
                    latest_delta_len = len(call_item.parameters)

                expected_call = json.dumps(
                    parser.detector.prev_tool_call_arr[index].get("arguments", {}),
                    ensure_ascii=False,
                )
                actual_call = parser.detector.streamed_args_for_tool[index]
                if latest_delta_len > 0:
                    actual_call = actual_call[:-latest_delta_len]
                remaining_call = expected_call.replace(actual_call, "", 1)
                call_item.parameters = remaining_call
                finish_reason_type = "tool_calls"

            tool_call = ToolCall(
                id=f"call_{base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode()}",
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=call_item.name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=(
                    None
                    if request.stream_options and request.stream_options.include_usage
                    else finish_reason_type
                ),
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
