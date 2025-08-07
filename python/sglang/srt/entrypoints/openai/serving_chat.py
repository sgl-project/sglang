import copy
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from openai_harmony import Message as OpenAIMessage

from sglang.srt.conversation import generate_chat_conv
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    get_system_message,
    parse_chat_input,
    parse_output_into_messages,
    render_for_completion,
)
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
    LogProbs,
    MessageProcessingResult,
    ToolCall,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_hidden_states_from_ret,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.jinja_template_utils import process_content_for_template_format
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.utils import convert_json_schema_to_str

logger = logging.getLogger(__name__)


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    def __init__(
        self, tokenizer_manager: TokenizerManager, template_manager: TemplateManager
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            from sglang.srt.function_call.harmony_tool_parser import (
                HarmonyToolCallParser,
            )

            self.harmony_tool_parser = HarmonyToolCallParser()

        # NOTE While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE: Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        if not self.use_harmony:
            processed_messages = self._process_messages(request, is_multimodal)

            # Build sampling parameters
            sampling_params = self._build_sampling_params(
                request,
                processed_messages.stop,
                processed_messages.tool_call_constraint,
            )

            # Handle single vs multiple requests
            if is_multimodal:
                prompt_kwargs = {"text": processed_messages.prompt}
            else:
                if isinstance(processed_messages.prompt_ids, str):
                    prompt_kwargs = {"text": processed_messages.prompt_ids}
                else:
                    prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

            adapted_request = GenerateReqInput(
                **prompt_kwargs,
                image_data=processed_messages.image_data,
                video_data=processed_messages.video_data,
                audio_data=processed_messages.audio_data,
                sampling_params=sampling_params,
                return_logprob=request.logprobs,
                logprob_start_len=-1,
                top_logprobs_num=request.top_logprobs or 0,
                stream=request.stream,
                return_text_in_logprobs=True,
                modalities=processed_messages.modalities,
                lora_path=request.lora_path,
                bootstrap_host=request.bootstrap_host,
                bootstrap_port=request.bootstrap_port,
                bootstrap_room=request.bootstrap_room,
                return_hidden_states=request.return_hidden_states,
                rid=request.rid,
            )
        else:
            processed_messages, prompt_ids = self._make_request_with_harmony(request)

            adapted_request = GenerateReqInput(
                input_ids=prompt_ids,
                sampling_params=self._build_sampling_params(
                    request,
                    request.stop,
                    tool_call_constraint=None,
                ),
                stream=request.stream,
                return_logprob=request.logprobs,
                logprob_start_len=-1,
                top_logprobs_num=request.top_logprobs or 0,
                return_text_in_logprobs=True,
                lora_path=request.lora_path,
                bootstrap_host=request.bootstrap_host,
                bootstrap_port=request.bootstrap_port,
                bootstrap_room=request.bootstrap_room,
                return_hidden_states=request.return_hidden_states,
                rid=request.rid,
            )

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        tool_call_constraint = None

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
            tool_call_constraint = parser.get_structure_constraint(request.tool_choice)

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
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

        stop = request.stop
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
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
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
            if self._get_enable_thinking_from_request(request):
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

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

    def _build_sampling_params(
        self,
        request: ChatCompletionRequest,
        stop: List[str],
        tool_call_constraint: Optional[Any],
    ) -> Dict[str, Any]:
        """Build sampling parameters for the request"""

        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens or request.max_completion_tokens,
            "min_new_tokens": request.min_tokens,
            "stop": stop,
            "stop_token_ids": request.stop_token_ids,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "regex": request.regex,
            "ebnf": request.ebnf,
            "n": request.n,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
            "logit_bias": request.logit_bias,
        }

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

        # Check if there are already existing output constraints
        has_existing_constraints = (
            sampling_params.get("regex")
            or sampling_params.get("ebnf")
            or sampling_params.get("structural_tag")
            or sampling_params.get("json_schema")
        )

        if tool_call_constraint and has_existing_constraints:
            logger.warning("Constrained decoding is not compatible with tool calls.")
        elif tool_call_constraint:
            constraint_type, constraint_value = tool_call_constraint
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
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat completion request"""
        return StreamingResponse(
            self._generate_chat_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}

        # Harmony tracking
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(request.n)
            ]

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                completion_tokens[index] = content["meta_info"]["completion_tokens"]
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    choice_logprobs = self._process_streaming_logprobs(
                        content, n_prev_tokens.get(index, 0)
                    )
                    n_prev_tokens[index] = len(
                        content["meta_info"]["output_token_logprobs"]
                    )

                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Process content delta
                if self.use_harmony:
                    harmony_parser = harmony_parsers[index]

                    new_token_ids = content["output_ids"]
                    for token_id in new_token_ids:
                        harmony_parser.process(token_id)

                    is_final = harmony_parser.current_channel == "final"
                    is_analysis = harmony_parser.current_channel == "analysis"
                    delta = harmony_parser.last_content_delta or ""

                    if is_analysis:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=delta),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        continue

                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(content=delta if delta else None),
                        finish_reason=None,
                        matched_stop=None,
                        logprobs=choice_logprobs,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    continue
                else:
                    stream_buffer = stream_buffers.get(index, "")
                    delta = content["text"][len(stream_buffer) :]
                    stream_buffers[index] = stream_buffer + delta

                # Handle reasoning content
                if (
                    self.tokenizer_manager.server_args.reasoning_parser
                    and request.separate_reasoning
                    and not self.use_harmony
                ):
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                if self.use_harmony and not is_final:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(reasoning_content=delta),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle tool calls
                # TODO: support tool call parsing for harmony
                if (
                    request.tool_choice != "none"
                    and request.tools
                    and not self.use_harmony
                ):
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                    ):
                        if chunk:
                            yield chunk

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type is not None and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk

                else:
                    # Regular content
                    if delta:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if request.stream_options and request.stream_options.include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        choices = []

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]
            output_ids = ret_item["output_ids"]

            if self.use_harmony:
                parser = parse_output_into_messages(output_ids)
                output_msgs = parser.messages
                if len(output_msgs) == 0:
                    # The generation has stopped during reasoning.
                    is_tool_call = False
                    reasoning_content = parser.current_content
                    final_content = None
                elif len(output_msgs) == 1:
                    # The generation has stopped during final message.
                    is_tool_call = False
                    reasoning_content = output_msgs[0].content[0].text
                    final_content = parser.current_content
                else:
                    if len(output_msgs) != 2:
                        raise ValueError(
                            "Expected 2 output messages (reasoning and final), "
                            f"but got {len(output_msgs)}."
                        )
                    reasoning_msg, final_msg = output_msgs
                    reasoning_content = reasoning_msg.content[0].text
                    final_content = final_msg.content[0].text
                    is_tool_call = final_msg.recipient is not None

                if is_tool_call:
                    # Extract tool call information from final message
                    tool_call = (
                        self.harmony_tool_parser.extract_tool_calls_from_message(
                            final_msg
                        )
                    )
                    tool_calls = [tool_call] if tool_call else []

                    message = ChatMessage(
                        role="assistant",
                        reasoning_content=reasoning_content,
                        content=None,  # Tool calls don't have regular content
                        tool_calls=tool_calls,
                    )
                else:
                    # Normal message
                    message = ChatMessage(
                        role="assistant",
                        reasoning_content=reasoning_content,
                        content=final_content,
                    )

                if is_tool_call:
                    finish_reason_type = "tool_calls"
                elif finish_reason:
                    finish_reason_type = (
                        finish_reason["type"] if finish_reason else "stop"
                    )
                else:
                    finish_reason_type = "stop"
                choice_data = ChatCompletionResponseChoice(
                    index=idx,
                    message=message,
                    logprobs=choice_logprobs,
                    finish_reason=finish_reason_type,
                    matched_stop=(
                        finish_reason["matched"]
                        if finish_reason and "matched" in finish_reason
                        else None
                    ),
                )
                choices.append(choice_data)
                continue

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=self.template_manager.force_reasoning,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if request.tool_choice != "none" and request.tools:
                tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
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
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
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

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
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
                        id=f"call_{uuid.uuid4().hex[:24]}",
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

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
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
                self.template_manager.force_reasoning,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_enable_thinking_from_request(self, request: ChatCompletionRequest) -> bool:
        """Extracts the 'enable_thinking' flag from request chat_template_kwargs.

        NOTE: This parameter is only useful for models that support enable_thinking
        flag, such as Qwen3.

        Args:
            request_obj: The request object (or an item from a list of requests).
        Returns:
            The boolean value of 'enable_thinking' if found, otherwise False.
        """
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is not None
        ):
            return request.chat_template_kwargs.get("enable_thinking")
        return False

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
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
                finish_reason=None,
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
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: FunctionCallParser,
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Only check if we have tool calls and the parser has tracked data
        if (
            not hasattr(parser.detector, "prev_tool_call_arr")
            or not parser.detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(parser.detector, "streamed_args_for_tool")
            or not parser.detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(parser.detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(parser.detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = parser.detector.prev_tool_call_arr[tool_index].get(
            "arguments", {}
        )
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = parser.detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None

    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
    ):
        messages: list[OpenAIMessage] = []

        # Add system message.
        # In Chat Completion API, browsing is enabled by default if the model
        # supports it.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        sys_msg = get_system_message(
            reasoning_effort=request.reasoning_effort,
            browser_description=None,
            python_description=None,
        )
        messages.append(sys_msg)

        # Add developer message.
        dev_msg = get_developer_message()
        messages.append(dev_msg)

        # Add user message.
        for chat_msg in request.messages:
            messages.append(parse_chat_input(chat_msg))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        return messages, prompt_token_ids
