from __future__ import annotations

import json
import logging
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.chat_input.validation import (
    validate_chat_input,
    validate_media_content,
)
from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.entrypoints.openai.chat_input_adapter import (
    create_chat_input_processor,
    from_chat_request,
    validate_chat_generation_options,
    with_prepared_options,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentTextPart,
    ChatCompletionMessageContentVideoPart,
    ChatCompletionMessageGenericParam,
    ChatCompletionMessageUserParam,
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
    PromptTokensDetails,
    SglExt,
    Tool,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    cached_tokens_details_from_dict,
    process_cached_tokens_details_from_ret,
    process_hidden_states_for_response,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    process_spec_tokens_details_from_ret,
    should_include_usage,
    spec_tokens_details_from_meta_info,
    to_openai_style_logprobs,
)
from sglang.srt.entrypoints.request_headers import apply_header_overrides
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.hunyuan_reasoning import (
    uses_hunyuan_reasoning_effort,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.runtime_context import get_serving
from sglang.srt.sampling.sampling_params import (
    set_request_reasoning_end_token_ids,
)
from sglang.srt.utils.weight_versions import build_endpoint_weight_version_metadata

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


def _extract_video_question(request: ChatCompletionRequest) -> Optional[str]:
    """Return text paired with a video in the last user turn."""
    for message in reversed(request.messages or []):
        if not isinstance(message, ChatCompletionMessageUserParam):
            continue
        content = message.content
        if not isinstance(content, list):
            continue
        has_video = any(
            isinstance(part, ChatCompletionMessageContentVideoPart) for part in content
        )
        if not has_video:
            continue
        return "".join(
            part.text
            for part in content
            if isinstance(part, ChatCompletionMessageContentTextPart)
        )
    return None


def _build_video_config(request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
    """Build request-scoped video processor config without model-specific fields."""
    config = dict(request.video_config or {})
    question = _extract_video_question(request)
    if question is not None:
        # Internal metadata derived from the message must not be overridden by
        # a model-specific public processor option.
        config["_question"] = question
    return config or None


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False
    _KIMI_K3_GENERATION_STUB_TOKENS = 3

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        input_processor=None,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.input_processor = (
            input_processor
            if input_processor is not None
            else create_chat_input_processor(tokenizer_manager, template_manager)
        )
        config = self.input_processor.config
        self.tool_call_parser = config.tool_call_parser
        self.reasoning_parser = config.reasoning_parser
        self._reasoning_detector = config.reasoning_detector
        self.chat_encoding_spec = config.chat_encoding_spec
        self.is_gpt_oss = config.is_gpt_oss
        self.is_gemma4 = config.is_gemma4

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _effective_tools(self, request: ChatCompletionRequest) -> List[Tool]:
        tools = list(request.tools or [])
        for message in request.messages:
            if (
                isinstance(message, ChatCompletionMessageGenericParam)
                and message.role in ("system", "developer")
                and message.tools
            ):
                tools.extend(message.tools)
        return tools

    def _decode_response(self, ret_item: Dict[str, Any]) -> Union[str, ErrorResponse]:
        """Extract text from response."""
        return ret_item["text"]

    def _get_parsed_response_fields(
        self,
        reasoning_text: Optional[str],
        tool_calls: Optional[List[Dict]],
    ) -> tuple[Optional[str], Optional[List[Dict]]]:
        """Post-process reasoning and tool_calls before building response."""
        return reasoning_text, tool_calls

    def _should_return_input_ids(self, request: ChatCompletionRequest) -> bool:
        """Whether prompt (input) token ids should be returned via sglext."""
        return request.return_input_ids_in_sglext or get_serving().return_input_ids

    def _should_return_output_ids(self, request: ChatCompletionRequest) -> bool:
        """Whether sampled output token ids should be returned via sglext."""
        return request.return_output_ids_in_sglext or get_serving().return_output_ids

    def _continuous_usage_cached_details(
        self, content: Dict[str, Any]
    ) -> Optional[PromptTokensDetails]:
        if not get_serving().enable_cache_report:
            return None
        return UsageProcessor._details_if_cached(
            content["meta_info"].get("cached_tokens", 0)
        )

    def _reported_prompt_tokens(self, meta_info: Dict[str, Any]) -> int:
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        if self.chat_encoding_spec == "kimi_k3":
            # K3's three-token assistant generation stub is model input, but the
            # reference API excludes it from billed/reported prompt tokens.
            prompt_tokens = max(0, prompt_tokens - self._KIMI_K3_GENERATION_STUB_TOKENS)
        return prompt_tokens

    async def _generate_stream_content(
        self,
        content: Dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        stream_offsets: Dict[int, int],
        reasoning_parser_dict: Dict,
        parser_dict: Dict,
        has_tool_calls: Dict[int, bool],
        choice_logprobs: Optional[Dict],
        finish_reason_type: Optional[str],
        continuous_usage_stats: bool,
        prompt_tokens: Dict[int, int],
        reasoning_tokens: Dict[int, int],
        completion_tokens: Dict[int, int],
    ) -> AsyncGenerator[str, None]:
        """Generate SSE chunks for streaming content."""
        offset = stream_offsets.get(index, 0)
        if get_serving().incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])

        # Attach logprobs to the first chunk emitted this step (reasoning,
        # tool-call, or content) so they aren't dropped when a parser is active
        # nor duplicated across chunks; flush any leftover at the end.
        remaining_logprobs = choice_logprobs

        # Handle reasoning content
        if self.reasoning_parser and request.separate_reasoning:
            reasoning_text, delta = self._process_reasoning_stream(
                index,
                delta,
                reasoning_parser_dict,
                content,
                request,
                finish_reason_type,
            )
            if reasoning_text:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    reasoning_content=reasoning_text,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Handle tool calls
        if self._tool_call_parsing_active(request):
            async for chunk in self._process_tool_call_stream(
                index,
                delta,
                parser_dict,
                content,
                request,
                has_tool_calls,
                continuous_usage_stats,
                flush=finish_reason_type is not None and finish_reason_type != "abort",
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
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                        cached_tokens=self._continuous_usage_cached_details(content),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    content=delta,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Flush logprobs still unattached this step — only when a parser is
        # active, since _process_tool_call_stream may consume the delta and emit
        # no content chunk. On the plain path an empty-delta step has no chunk
        # to attach to either way, and a standalone empty-delta logprobs chunk
        # is not a shape clients expect.
        if remaining_logprobs is not None and (
            self.reasoning_parser or self.tool_call_parser
        ):
            usage = None
            if continuous_usage_stats:
                usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens.get(index, 0),
                    reasoning_tokens=reasoning_tokens.get(index, 0),
                    completion_tokens=completion_tokens.get(index, 0),
                    cached_tokens=self._continuous_usage_cached_details(content),
                ).model_dump()

            yield build_sse_content(
                chunk_id=content["meta_info"]["id"],
                created=int(time.time()),
                model=request.model,
                index=index,
                logprobs=remaining_logprobs,
                usage=usage,
            )

    def _tool_call_parsing_active(self, request: ChatCompletionRequest) -> bool:
        """Whether this request's output runs through the tool-call parser.

        The reasoning parser is told the same thing, so channel-framed formats
        keep their framing intact exactly when a tool-call parser consumes it.
        """
        return bool(
            request.tool_choice != "none"
            and self._effective_tools(request)
            and self.tool_call_parser
        )

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        chat_input = from_chat_request(request)
        return (
            validate_chat_input(chat_input)
            or validate_media_content(
                chat_input, self.tokenizer_manager.model_config.is_multimodal
            )
            or validate_chat_generation_options(request)
        )

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:

        request = request.model_copy(deep=True)

        # Header-based opt-in (same rationale as request_headers.py).
        if raw_request is not None and not request.return_input_ids_in_sglext:
            if raw_request.headers.get("x-sglext-return-input-ids") == "1":
                request.return_input_ids_in_sglext = True

        if raw_request is not None and not request.return_output_ids_in_sglext:
            if raw_request.headers.get("x-sglext-return-output-ids") == "1":
                request.return_output_ids_in_sglext = True

        reasoning_effort = None
        if not uses_hunyuan_reasoning_effort(
            self.reasoning_parser, self.template_manager.reasoning_config
        ):
            reasoning_effort = (
                request.chat_template_kwargs.pop("reasoning_effort", None)
                if request.chat_template_kwargs
                else None
            )

        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        if request.stream:
            if request.return_prompt_token_ids:
                raise ValueError(
                    "return_prompt_token_ids is not supported with streaming. "
                    "Please set stream=false when using return_prompt_token_ids=true."
                )
            if request.return_token_ids:
                raise ValueError(
                    "return_token_ids is not supported with streaming on "
                    "/v1/chat/completions. Please set stream=false when using "
                    "return_token_ids=true."
                )
            if request.return_meta_info:
                raise ValueError(
                    "return_meta_info is not supported with streaming. "
                    "Please set stream=false when using return_meta_info=true."
                )

        processed_messages = self.input_processor.prepare(from_chat_request(request))
        request = with_prepared_options(request, processed_messages)
        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
            renderer_handles_response_format=self.chat_encoding_spec == "kimi_k3",
        )
        set_request_reasoning_end_token_ids(
            sampling_params, processed_messages.reasoning_end_token_ids
        )

        prompt_kwargs = processed_messages.prompt.to_generate_kwargs()

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            return_sampling_mask=request.return_sampling_mask,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=request.extra_key,
            cache_salt=request.cache_salt,
            require_reasoning=processed_messages.require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
            video_config=_build_video_config(request),
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
            use_audio_in_video=getattr(request, "use_audio_in_video", False),
            return_prompt_token_ids=(
                request.return_prompt_token_ids
                or request.return_token_ids
                or self._should_return_input_ids(request)
            ),
        )
        if (
            raw_request is not None
            and envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get()
        ):
            apply_header_overrides(adapted_request, raw_request.headers)

        return adapted_request, request

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
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
        stream_offsets = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}

        # Usage tracking
        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        cached_tokens_details = {}
        spec_tokens_details = {}
        image_tokens = {}
        audio_tokens = {}
        video_tokens = {}
        input_ids: Optional[List[int]] = None
        output_ids: Dict[int, List[int]] = {}

        stream_started = False
        error_aborted = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                get_serving().stream_response_default_include_usage,
            )

            return_input_ids = self._should_return_input_ids(request)
            return_output_ids = self._should_return_output_ids(request)

            ids_framed = (
                raw_request is not None
                and raw_request.headers.get("x-sglext-ids-framed") == "1"
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = self._reported_prompt_tokens(
                    content["meta_info"]
                )
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)
                cached_tokens_details[index] = content["meta_info"].get(
                    "cached_tokens_details", None
                )
                if request.return_spec_tokens_details:
                    spec_tokens_details[index] = spec_tokens_details_from_meta_info(
                        content["meta_info"]
                    )
                image_tokens[index] = content["meta_info"].get("image_tokens", 0)
                audio_tokens[index] = content["meta_info"].get("audio_tokens", 0)
                video_tokens[index] = content["meta_info"].get("video_tokens", 0)

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                if return_input_ids and input_ids is None:
                    # The prompt is the full, shared prompt (same across choices
                    # and constant across chunks), so capture it once.
                    chunk_input_ids = content.get("prompt_token_ids")
                    if chunk_input_ids is not None:
                        input_ids = list(chunk_input_ids)

                if return_output_ids:
                    chunk_output_ids = content.get("output_ids")
                    if chunk_output_ids is not None:
                        if get_serving().incremental_streaming_output:
                            accumulated = output_ids.setdefault(index, [])
                            if finish_reason_type == "abort":
                                # The abort chunk re-sends the last token plus any coalesced deltas;
                                # keep only what brings the total up to completion_tokens.
                                keep = completion_tokens[index] - len(accumulated)
                                chunk_output_ids = chunk_output_ids[: max(keep, 0)]
                            accumulated.extend(chunk_output_ids)
                        else:
                            # Intermediate chunks share the live state.output_ids
                            # list; the final chunk is a stable copy.
                            output_ids[index] = chunk_output_ids

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        ).model_dump()
                    n_prev_tokens[index] = total_output_logprobs

                # Track finish_reason for each index
                if finish_reason_type:
                    # Abort with an explicit error status_code is a system error
                    # (timeout, OOM, validation): emit a streaming error chunk.
                    # A graceful abort (no status_code, e.g. user-initiated via
                    # /abort_request or session lifecycle cleanup) falls through
                    # to the normal chunk path, matching the non-stream behavior
                    # in tokenizer_manager._handle_abort_finish_reason.
                    if finish_reason_type == "abort" and isinstance(
                        finish_reason.get("status_code"), HTTPStatus
                    ):
                        code = finish_reason["status_code"]
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        error_aborted = True
                        break
                    finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    yield build_sse_content(
                        chunk_id=content["meta_info"]["id"],
                        created=int(time.time()),
                        model=request.model,
                        index=index,
                        role="assistant",
                        content="",
                    )
                    stream_started = True

                # Generate streaming content (override in subclass for custom behavior)
                async for chunk in self._generate_stream_content(
                    content=content,
                    index=index,
                    request=request,
                    stream_offsets=stream_offsets,
                    reasoning_parser_dict=reasoning_parser_dict,
                    parser_dict=parser_dict,
                    has_tool_calls=has_tool_calls,
                    choice_logprobs=choice_logprobs,
                    finish_reason_type=finish_reason_type,
                    continuous_usage_stats=continuous_usage_stats,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                ):
                    yield chunk

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                matched_stop = finish_reason_data.get("matched")
                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=idx,
                    finish_reason=final_finish_reason,
                    matched_stop=matched_stop,
                )

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        response_hidden_states = process_hidden_states_for_response(
                            choice_hidden_states, request.return_hidden_states
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=response_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            sglext_routed = None
            if request.return_routed_experts and routed_experts:
                sglext_routed = next(
                    (v for v in routed_experts.values() if v is not None), None
                )

            sglext_cached_tokens_details = None
            if request.return_cached_tokens_details and cached_tokens_details:
                first_details = next(
                    (v for v in cached_tokens_details.values() if v is not None), None
                )
                if first_details is not None:
                    sglext_cached_tokens_details = cached_tokens_details_from_dict(
                        first_details
                    )

            sglext_spec_tokens_details = None
            if request.return_spec_tokens_details and spec_tokens_details:
                spec_details = [
                    spec_tokens_details[index]
                    for index in sorted(spec_tokens_details)
                    if spec_tokens_details[index] is not None
                ]
                if spec_details:
                    sglext_spec_tokens_details = (
                        spec_details if request.n > 1 else spec_details[0]
                    )

            # Omit token ids after an error abort.
            sglext_input_ids = None
            if return_input_ids and input_ids and not error_aborted:
                sglext_input_ids = list(input_ids)

            sglext_output_ids = None
            if return_output_ids and output_ids and not error_aborted:
                sglext_output_ids = [
                    list(output_ids.get(i, [])) for i in range(request.n)
                ]

            sglext_full = SglExt(
                routed_experts=sglext_routed,
                cached_tokens_details=sglext_cached_tokens_details,
                spec_tokens_details=sglext_spec_tokens_details,
                input_ids=sglext_input_ids,
                output_ids=sglext_output_ids,
            )
            sglext_non_ids, sglext_ids = sglext_full.split_ids()

            if ids_framed:
                # A named SSE event lets transit hops pick the ids out without parsing JSON;
                # the other sglext fields keep the plain data-chunk shape.
                if sglext_non_ids is not None:
                    sglext_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],
                        model=request.model,
                        sglext=sglext_non_ids,
                    )
                    yield f"data: {sglext_chunk.model_dump_json()}\n\n"
                if sglext_ids is not None:
                    sglext_ids_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],
                        model=request.model,
                        sglext=sglext_ids,
                    )
                    yield f"event: sglext_ids\ndata: {sglext_ids_chunk.model_dump_json()}\n\n"
            elif sglext_non_ids is not None or sglext_ids is not None:
                sglext_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # sglext is at response level
                    model=request.model,
                    sglext=sglext_full,
                )
                yield f"data: {sglext_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if include_usage:
                # Multimodal tokens are per-prompt (input side), so aggregate
                # once per prompt (first choice), matching prompt/cached semantics.
                total_image_tokens = sum(
                    tok for idx, tok in image_tokens.items() if idx % request.n == 0
                )
                total_audio_tokens = sum(
                    tok for idx, tok in audio_tokens.items() if idx % request.n == 0
                )
                total_video_tokens = sum(
                    tok for idx, tok in video_tokens.items() if idx % request.n == 0
                )
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=get_serving().enable_cache_report,
                    image_tokens=total_image_tokens,
                    audio_tokens=total_audio_tokens,
                    video_tokens=total_video_tokens,
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
            if not stream_started:
                raise
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
        if self.chat_encoding_spec == "kimi_k3":
            ret = [
                {
                    **item,
                    "meta_info": {
                        **item["meta_info"],
                        "prompt_tokens": self._reported_prompt_tokens(
                            item["meta_info"]
                        ),
                    },
                }
                for item in ret
            ]

        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = (
            None
            if request.return_meta_info
            else process_routed_experts_from_ret(first_ret, request)
        )
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        spec_details = [
            detail
            for detail in (
                process_spec_tokens_details_from_ret(item, request) for item in ret
            )
            if detail is not None
        ]
        spec_tokens_details = (
            spec_details
            if request.n > 1
            else (spec_details[0] if spec_details else None)
        )
        input_ids = None
        if self._should_return_input_ids(request) and "prompt_token_ids" in ret[0]:
            input_ids = list(ret[0]["prompt_token_ids"])
        output_ids = None
        if self._should_return_output_ids(request):
            output_ids = [list(ret_item["output_ids"]) for ret_item in ret]
        response_sglext = None
        if (
            routed_experts
            or cached_tokens_details
            or spec_tokens_details
            or input_ids is not None
            or output_ids is not None
        ):
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
                spec_tokens_details=spec_tokens_details,
                input_ids=input_ids,
                output_ids=output_ids,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]

            text = self._decode_response(ret_item)
            if isinstance(text, ErrorResponse):
                return ORJSONResponse(content=text.model_dump(), status_code=text.code)

            # Handle reasoning content
            reasoning_text = None
            if self.reasoning_parser and request.separate_reasoning:
                force_reasoning = (
                    self.template_manager.force_reasoning
                    or self.input_processor._get_reasoning_from_request(
                        from_chat_request(request)
                    )
                )
                try:
                    parser = ReasoningParser(
                        model_type=self.reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=force_reasoning,
                        request=request,
                        tokenizer=self.tokenizer_manager.tokenizer,
                        tool_call_parser_active=self._tool_call_parsing_active(request),
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
            effective_tools = self._effective_tools(request)
            if self._tool_call_parsing_active(request):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    effective_tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            # Extract prompt_token_ids if requested
            choice_prompt_token_ids = (
                ret_item.get("prompt_token_ids")
                if request.return_prompt_token_ids or request.return_token_ids
                else None
            )
            choice_token_ids = (
                ret_item["output_ids"] if request.return_token_ids else None
            )

            choice_meta_info = (
                ret_item["meta_info"] if request.return_meta_info else None
            )
            # NOTE: content should not be None but empty string to make sure retokenize consistency.
            reasoning_text, tool_calls = self._get_parsed_response_fields(
                reasoning_text, tool_calls
            )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else "",
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
                prompt_token_ids=choice_prompt_token_ids,
                response_token_ids=choice_token_ids,
                meta_info=choice_meta_info,
            )
            choices.append(choice_data)

        # Calculate usage. Multimodal tokens are per-prompt (input side), so
        # aggregate once per prompt (stride by n), matching prompt/cached semantics.
        image_tokens = sum(
            ret[i]["meta_info"].get("image_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        audio_tokens = sum(
            ret[i]["meta_info"].get("audio_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        video_tokens = sum(
            ret[i]["meta_info"].get("video_tokens", 0)
            for i in range(0, len(ret), request.n)
        )
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=get_serving().enable_cache_report,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata=build_endpoint_weight_version_metadata(ret[0]["meta_info"]),
            sglext=response_sglext,
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

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        if self.tool_call_parser == "kimi_k3":
            return f"{call_item.name}:{history_tool_calls_cnt + call_item.tool_index}"
        if self.tool_call_parser != "kimi_k2":
            # A simple uuid is sufficient for all models except for Kimi-K2.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        tool_call_id = (
            f"functions.{call_item.name}:"
            f"{history_tool_calls_cnt + call_item.tool_index}"
        )
        logger.debug(
            f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
        )
        return tool_call_id

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response"""

        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)

        # Try model-specific parser when output is in native format.
        # For required/named: only use parser when structural_tag was used
        # as constraint (mirrors the streaming path). For auto: always try.
        if self.tool_call_parser:
            parser = FunctionCallParser(
                tools, self.tool_call_parser, tokenizer=self.tokenizer_manager.tokenizer
            )
            detector_owns_format = (
                parser.detector.supports_structural_tag()
                or parser.detector.parses_required_natively()
            )
            should_try_parser = not is_required or detector_owns_format
            if should_try_parser and parser.has_tool_call(text):
                try:
                    text, call_info_list = parser.parse_non_stream(text)
                    if not call_info_list:
                        logger.warning(
                            "Tool call marker present but no complete call parsed "
                            "from %s output; dropping the incomplete call",
                            self.tool_call_parser,
                        )
                        logger.debug(
                            "Unparsed tool call output (%d chars): %r",
                            len(text),
                            text[:2000],
                        )
                        return ToolCallProcessingResult(None, text, finish_reason)

                    tool_calls = []
                    for call_info in call_info_list:
                        tool_id = self._process_tool_call_id(
                            call_info, history_tool_calls_cnt
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                index=getattr(call_info, "tool_index", None),
                                function=FunctionResponse(
                                    name=call_info.name,
                                    arguments=call_info.parameters,
                                ),
                            )
                        )
                    if finish_reason["type"] == "stop":
                        finish_reason["type"] = "tool_calls"
                        finish_reason["matched"] = None
                    return ToolCallProcessingResult(tool_calls, text, finish_reason)
                except Exception as e:
                    logger.error(f"Tool call parsing error: {e}")
                    return ToolCallProcessingResult(None, text, finish_reason)

            if is_required and detector_owns_format:
                logger.warning(
                    "Required tool call missing from %s output (%d chars)",
                    self.tool_call_parser,
                    len(text),
                )
                logger.debug("Unparsed required tool call output: %r", text[:2000])
                return ToolCallProcessingResult(None, text, finish_reason)

        # json_schema constraint → JSON array output for required/named
        if is_required:
            original_finish_type = finish_reason["type"]
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                tool_call_data = orjson.loads(text)
                if isinstance(tool_call_data, dict):
                    tool_call_data = [tool_call_data]
                if not isinstance(tool_call_data, list):
                    raise ValueError(
                        "expected a JSON array of tool calls, got "
                        f"{type(tool_call_data).__name__}"
                    )
                if not all(
                    isinstance(tool, dict) and "name" in tool for tool in tool_call_data
                ):
                    raise ValueError(
                        "every tool call must be a JSON object with a 'name'"
                    )
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    parameters = json.dumps(
                        tool.get("parameters", {}), ensure_ascii=False
                    )
                    call_info = ToolCallItem(
                        tool_index=i,
                        name=tool["name"],
                        parameters=parameters,
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=parameters,
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                logger.debug("Unparsed required tool call output: %r", text[:2000])
                finish_reason["type"] = original_finish_type
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        output_token_logprobs = content["meta_info"]["output_token_logprobs"]
        output_top_logprobs = content["meta_info"].get("output_top_logprobs", [])
        if not get_serving().incremental_streaming_output:
            output_token_logprobs = output_token_logprobs[
                n_prev_token:total_output_logprobs
            ]
            output_top_logprobs = output_top_logprobs[
                n_prev_token:total_output_logprobs
            ]
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=output_token_logprobs,
            output_top_logprobs=output_top_logprobs,
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
        finish_reason_type: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self.input_processor._get_reasoning_from_request(
                    from_chat_request(request)
                )
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
                tokenizer=self.tokenizer_manager.tokenizer,
                tool_call_parser_active=self._tool_call_parsing_active(request),
            )
        reasoning_parser = reasoning_parser_dict[index]
        reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(delta)
        if finish_reason_type is not None and finish_reason_type != "abort":
            end_reasoning_text, end_normal_text = reasoning_parser.parse_stream_end()
            if end_reasoning_text:
                reasoning_text = (reasoning_text or "") + end_reasoning_text
            if end_normal_text:
                normal_text = (normal_text or "") + end_normal_text
        return reasoning_text, normal_text

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
        return idx

    def supports_native_reasoning_history(self) -> bool:
        """Whether the chat encoder takes history as ``reasoning_content`` rather
        than via :meth:`wrap_reasoning_history`; see
        :func:`chat_encoding.spec_owns_reasoning_history` for why.
        """
        return chat_encoding.spec_owns_reasoning_history(self.chat_encoding_spec)

    def wrap_reasoning_history(self, reasoning_text: str) -> str:
        """Wrap prior-turn reasoning in the detector's own start/end tokens.

        Pulling the delimiters from the detector keeps adapters in lockstep
        with any future parser that ships non-``<think>`` markers — Mistral's
        ``[THINK]``, Gemma4's ``think_start_self_label = "thought\\n"``, etc.
        Falling back to a plain string is unsafe: it would let prior
        thinking text reach a non-reasoning model as ordinary assistant
        content, so the caller must surface this state, not paper over it.
        """
        if self._reasoning_detector is None:
            raise ValueError(
                "Cannot rewrap thinking history: no reasoning detector is "
                "configured for this model"
            )
        d = self._reasoning_detector
        return (
            f"{d.think_start_token}{d.think_start_self_label}"
            f"{reasoning_text}\n{d.think_end_token}"
        )

    def _reasoning_default_mode(self) -> Optional[str]:
        if self._reasoning_detector is None:
            return None
        return self._reasoning_detector.reasoning_default

    def _get_reasoning_toggle_param(self) -> Optional[str]:
        """Resolve the chat-template kwarg that toggles reasoning, if any."""
        config = self.template_manager.reasoning_config
        if config is not None:
            return config.toggle_param

        mode = self._reasoning_default_mode()
        if mode in ("thinking", "enable_thinking"):
            return mode
        if mode in ("explicit_thinking", "explicit_enable_thinking"):
            return mode.replace("explicit_", "")
        return None

    def apply_reasoning_enabled(
        self, request: ChatCompletionRequest, enabled: bool
    ) -> None:
        """Force the request into the requested reasoning-on/off mode.

        Mirrors the read-side logic in ``_get_reasoning_from_request``;
        the two must stay in sync. Always-on models cannot be disabled,
        so explicit ``enabled=False`` raises rather than silently leaving
        reasoning on.
        """
        if not self.reasoning_parser:
            if enabled:
                raise ValueError(
                    "Anthropic thinking is not supported for models without "
                    "a reasoning parser"
                )
            return

        if self.reasoning_parser == "hunyuan":
            config = self.template_manager.reasoning_config
            if config is not None and config.special_case == "hunyuan_effort":
                request.reasoning_effort = "high" if enabled else "no_think"
            else:
                request.reasoning_effort = "medium" if enabled else "no_think"
            return

        if self.reasoning_parser == "inkling":
            # Effort-conditioned, not toggled: "none" (0.0) is the off switch.
            if not enabled:
                request.reasoning_effort = "none"
                return

        config = self.template_manager.reasoning_config
        is_mistral = (config is not None and config.special_case == "mistral") or (
            config is None and self._reasoning_default_mode() == "mistral"
        )
        if is_mistral:
            request.reasoning_effort = "medium" if enabled else "none"
            return

        is_always_on = (config is not None and config.special_case == "always") or (
            config is None and self._reasoning_default_mode() == "always"
        )
        if is_always_on:
            if not enabled:
                raise ValueError(
                    f"Reasoning parser '{self.reasoning_parser}' is always-on "
                    f"and cannot be disabled via Anthropic thinking"
                )
            return

        toggle_param = self._get_reasoning_toggle_param()
        # The read side (``_get_reasoning_from_request``) returns False
        # whenever ``config.toggle_param is None`` OR
        # ``config.default_enabled is None``. The write side must mirror
        # both conditions: if ``default_enabled`` is unset we cannot
        # actually honor an ``enabled=True`` request even when the toggle
        # name itself is resolvable, so writing the kwarg would set up the
        # template to emit reasoning tokens while the parser ignores them
        # (literal ``<think>`` markers leak into the assistant text).
        config = self.template_manager.reasoning_config
        read_side_supported = toggle_param is not None and (
            config is None or config.default_enabled is not None
        )
        if not read_side_supported:
            if not enabled:
                return
            raise ValueError(
                f"Anthropic thinking is not supported for reasoning parser "
                f"'{self.reasoning_parser}'"
            )

        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        chat_template_kwargs[toggle_param] = enabled
        request.chat_template_kwargs = chat_template_kwargs

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
        flush: bool = False,
    ):
        """Process tool calls in streaming response.

        With flush=True (the terminal delta), the parser also drains text it
        held back waiting for a marker that can no longer arrive.
        """
        effective_tools = self._effective_tools(request)
        if index not in parser_dict:
            is_required = request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            )
            # For required/named tool choice: use JsonArrayParser when the
            # constrained output is plain JSON (detector doesn't support
            # structural_tag or no parser configured). Use FunctionCallParser
            # only when the detector supports structural_tag and will produce
            # native format output.
            if is_required:
                use_native_parser = False
                if self.tool_call_parser:
                    probe = FunctionCallParser(
                        tools=effective_tools,
                        tool_call_parser=self.tool_call_parser,
                        tokenizer=self.tokenizer_manager.tokenizer,
                    )
                    use_native_parser = (
                        probe.detector.supports_structural_tag()
                        or probe.detector.parses_required_natively()
                    )
                if use_native_parser:
                    parser_dict[index] = probe
                else:
                    parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=effective_tools,
                    tool_call_parser=self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )

        parser = parser_dict[index]

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, effective_tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)
            if flush:
                end_text, end_calls = parser.parse_stream_end()
                normal_text = (normal_text or "") + end_text
                calls = list(calls) + end_calls

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

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
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

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = self._reported_prompt_tokens(content["meta_info"])
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cached_tokens=self._continuous_usage_cached_details(content),
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        if isinstance(expected_args, str):
            expected_call = expected_args
        else:
            expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call[len(actual_call) :]
            if expected_call.startswith(actual_call)
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
