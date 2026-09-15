from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.output_padding import OutputPaddingPlan
from sglang.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    SglExt,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
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
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.code_completion_parser import (
    generate_completion_prompt_from_request,
)
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils.weight_versions import build_endpoint_weight_version_metadata
from sglang.utils import convert_json_schema_to_str

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class OpenAIServingCompletion(OpenAIServingBase):
    """Handler for /v1/completion requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "cmpl-"

    def _validate_request(self, request: CompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        prompt = request.prompt
        if not prompt or (isinstance(prompt, list) and all(not p for p in prompt)):
            return "Prompt cannot be empty"

        return self._validate_padded_output_budget(request)

    def _validate_padded_output_budget(
        self, request: CompletionRequest
    ) -> Optional[str]:
        target = get_serving().padded_output_tokens
        if target is None:
            return None
        # Strictly greater, not >=: the emitted run is clamped at max_new_tokens
        # inclusively, so a budget of exactly the target can never overrun it.
        if request.max_tokens > target:
            return (
                f"max_tokens ({request.max_tokens}) exceeds --padded-output-tokens "
                f"({target}). This server pads every completion to exactly {target} "
                "tokens and cannot admit a request that could outrun that."
            )
        if self._pad_token_id() is None:
            return (
                "--padded-output-tokens is set but this model exposes no "
                "eos_token_id to pad with, so output padding cannot be applied."
            )
        if request.echo:
            # Echo replays the prompt through the completion stream, which the
            # padding does not cover, and with logprobs it prepends the input
            # positions into the payload the per-token split consumes.
            return (
                "echo is not supported with --padded-output-tokens: the echoed "
                "prompt is not padded and would carry the prompt's own length."
            )
        return None

    def _pad_token_id(self) -> Optional[int]:
        # hf_eos_token_id is a set, whose iteration order is arbitrary; min pins
        # the pad id so it does not vary between restarts.
        eos_ids = self.tokenizer_manager.model_config.hf_eos_token_id
        if not eos_ids:
            return None
        return min(eos_ids)

    def _build_padding_plan(
        self, request: CompletionRequest
    ) -> Optional[OutputPaddingPlan]:
        target = get_serving().padded_output_tokens
        if target is None:
            return None
        pad_token_id = self._pad_token_id()
        if pad_token_id is None:
            raise ValueError(
                "--padded-output-tokens is set but this model exposes no eos_token_id."
            )
        return OutputPaddingPlan.build(
            target_tokens=target,
            pad_token_id=pad_token_id,
            pad_token_text=self.tokenizer_manager.tokenizer.decode([pad_token_id]),
            return_logprob=request.logprobs is not None,
            top_logprobs_width=request.logprobs or 0,
        )

    def _convert_to_internal_request(
        self,
        request: CompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, CompletionRequest]:
        """Convert OpenAI completion request to internal format"""
        # NOTE: with openai API, the prompt's logprobs are always not computed
        if request.echo and request.logprobs:
            logger.warning(
                "Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use the native /generate API."
            )
        # Process prompt
        prompt = request.prompt
        if self.template_manager.completion_template_name is not None:
            prompt = generate_completion_prompt_from_request(request)

        # Set logprob start length based on echo and logprobs
        if request.echo and request.logprobs:
            logprob_start_len = 0
        else:
            logprob_start_len = -1

        # Build sampling parameters
        sampling_params = self._build_sampling_params(request)

        # Determine prompt format
        if isinstance(prompt, str) or (
            isinstance(prompt, list) and isinstance(prompt[0], str)
        ):
            prompt_kwargs = {"text": prompt}
        else:
            prompt_kwargs = {"input_ids": prompt}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            sampling_params=sampling_params,
            return_logprob=request.logprobs is not None,
            top_logprobs_num=request.logprobs if request.logprobs is not None else 0,
            logprob_start_len=logprob_start_len,
            return_text_in_logprobs=True,
            stream=request.stream,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            return_prompt_token_ids=request.return_token_ids,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=request.extra_key,
            cache_salt=request.cache_salt,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
        )

        return adapted_request, request

    def _build_sampling_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        # Start with common parameters
        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "min_new_tokens": request.min_tokens,
            "stop": request.stop,
            "stop_token_ids": request.stop_token_ids,
            "stop_regex": request.stop_regex,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "regex": request.regex,
            "json_schema": request.json_schema,
            "ebnf": request.ebnf,
            "n": request.n,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
            "logit_bias": request.logit_bias,
            "custom_params": request.custom_params,
            "sampling_seed": request.seed,
        }

        # Handle response_format constraints
        if request.response_format and request.response_format.type == "json_schema":
            json_schema = request.response_format.json_schema
            schema = getattr(json_schema, "schema_", None)
            if schema is None:
                raise ValueError(
                    "schema_ is required for json_schema response format request."
                )
            sampling_params["json_schema"] = convert_json_schema_to_str(schema)
        elif request.response_format and request.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif (
            request.response_format and request.response_format.type == "structural_tag"
        ):
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                request.response_format.model_dump(by_alias=True)
            )

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming completion request"""
        generator = self._generate_completion_stream(
            adapted_request, request, raw_request
        )

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
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

    def _stream_frame(
        self,
        *,
        request: CompletionRequest,
        response_id: str,
        created: int,
        choice: CompletionResponseStreamChoice,
        usage: Optional[Any] = None,
    ) -> str:
        chunk = CompletionStreamResponse(
            id=response_id,
            created=created,
            object="text_completion",
            choices=[choice],
            model=request.model,
        )
        if usage is not None:
            chunk.usage = usage
        return f"data: {chunk.model_dump_json()}\n\n"

    def _chunk_token_count(self, content: Dict[str, Any], already_framed: int) -> int:
        output_ids = content.get("output_ids")
        if output_ids is None:
            return max(
                0, content["meta_info"].get("completion_tokens", 0) - already_framed
            )
        if get_serving().incremental_streaming_output:
            return len(output_ids)
        return max(0, len(output_ids) - already_framed)

    def _pad_run_frames(
        self,
        *,
        request: CompletionRequest,
        response_id: str,
        created: int,
        index: int,
        padding: OutputPaddingPlan,
        wire_tokens: Dict[int, int],
        manager_tokens: int,
        prompt_tokens: int,
        reasoning_tokens: int,
        continuous_usage_stats: bool,
        carry_text: str = "",
        carry_prompt_token_ids: Optional[List[int]] = None,
    ) -> Iterator[str]:
        framed = wire_tokens[index]
        # Unreachable while the budget check stands, since that admits only
        # max_tokens <= target. Reaching it means the bound broke, so refuse to
        # report a padded count for a completion that outran the target. The real
        # frames are already on the wire by now, so what this refuses is the false
        # count, not the leak itself.
        if max(framed, manager_tokens) > padding.target_tokens:
            raise RuntimeError(
                f"completion outran --padded-output-tokens: {framed} tokens framed "
                f"and manager reported {manager_tokens}, target "
                f"{padding.target_tokens}"
            )

        def usage_for(count: int) -> Optional[Any]:
            if not continuous_usage_stats:
                return None
            return UsageProcessor.calculate_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=count,
                reasoning_tokens=reasoning_tokens,
            )

        for pad_offset in range(padding.pad_count(framed)):
            wire_tokens[index] += 1
            first = pad_offset == 0
            yield self._stream_frame(
                request=request,
                response_id=response_id,
                created=created,
                choice=CompletionResponseStreamChoice(
                    index=index,
                    # Pad frames add no text of their own: the run moves the token
                    # and frame counts, and emitting the terminal token's text
                    # would change the completion the client reassembles.
                    text=(carry_text if first else ""),
                    logprobs=(
                        padding.filler_logprobs() if padding.return_logprob else None
                    ),
                    finish_reason=None,
                    matched_stop=None,
                    token_ids=(
                        [padding.pad_token_id] if request.return_token_ids else None
                    ),
                    prompt_token_ids=(carry_prompt_token_ids if first else None),
                ),
                usage=usage_for(wire_tokens[index]),
            )
        if padding.pad_count(framed):
            carry_text, carry_prompt_token_ids = "", None

        # Unpadded, the terminal frame is the last thing that tracks the verdict:
        # a completion ending on its own token stops with "stop" and names the
        # matched sequence, one filling the budget stops with "length". Every
        # padded completion emits exactly target_tokens, so "length" is constant
        # and true.
        yield self._stream_frame(
            request=request,
            response_id=response_id,
            created=created,
            choice=CompletionResponseStreamChoice(
                index=index,
                text=carry_text,
                logprobs=None,
                finish_reason="length",
                matched_stop=None,
                prompt_token_ids=carry_prompt_token_ids,
            ),
            usage=usage_for(wire_tokens[index]),
        )

    async def _generate_completion_stream(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion response"""
        created = int(time.time())

        # State tracking for streaming
        stream_offsets = {}
        n_prev_tokens = {}
        n_prev_token_ids = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        reasoning_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        cached_tokens_details = {}
        spec_tokens_details = {}

        padding = self._build_padding_plan(request)
        # Tokens actually framed onto the wire, per index. Distinct from
        # completion_tokens, which mirrors whatever the manager reports and can
        # run ahead of what this loop emitted; the pad count must be derived
        # from the emitted figure or the padded total drifts off the target.
        wire_tokens = {}
        # Chunk-scoped payload from a chunk that carried no new token. The offsets
        # feeding it have already advanced, so it has to ride the next frame or it
        # is lost; a zero-token completion has no next real frame, only pads.
        pending_text = {}
        pending_prompt_ids = {}
        aborted = False

        stream_started = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                get_serving().stream_response_default_include_usage,
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                text = content["text"]
                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
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

                is_first_chunk = index not in stream_offsets
                offset = stream_offsets.get(index, 0)
                # Handle echo for first chunk
                if is_first_chunk:  # The first chunk
                    if request.echo:
                        echo_text = self._get_echo_text(request, index)
                        text = echo_text + text

                # Handle logprobs
                logprobs = None
                if request.logprobs is not None:
                    # The first chunk and echo is enabled.
                    if is_first_chunk and request.echo:
                        input_token_logprobs = content["meta_info"][
                            "input_token_logprobs"
                        ]
                        input_top_logprobs = content["meta_info"]["input_top_logprobs"]
                    else:
                        input_token_logprobs = None
                        input_top_logprobs = None

                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if (
                        n_prev_token < total_output_logprobs
                        or input_token_logprobs is not None
                    ):
                        output_token_logprobs = content["meta_info"][
                            "output_token_logprobs"
                        ]
                        output_top_logprobs = content["meta_info"].get(
                            "output_top_logprobs", []
                        )
                        if not get_serving().incremental_streaming_output:
                            output_token_logprobs = output_token_logprobs[
                                n_prev_token:total_output_logprobs
                            ]
                            output_top_logprobs = output_top_logprobs[
                                n_prev_token:total_output_logprobs
                            ]
                        logprobs = to_openai_style_logprobs(
                            input_token_logprobs=input_token_logprobs,
                            input_top_logprobs=input_top_logprobs,
                            output_token_logprobs=output_token_logprobs,
                            output_top_logprobs=output_top_logprobs,
                        )
                    n_prev_tokens[index] = total_output_logprobs

                chunk_token_ids = None
                chunk_prompt_token_ids = None
                if request.return_token_ids:
                    output_ids = content["output_ids"]
                    if not get_serving().incremental_streaming_output:
                        n_prev_token_id = n_prev_token_ids.get(index, 0)
                        chunk_token_ids = output_ids[n_prev_token_id:]
                        n_prev_token_ids[index] = len(output_ids)
                    else:
                        chunk_token_ids = output_ids
                    if is_first_chunk:
                        chunk_prompt_token_ids = content.get("prompt_token_ids")

                # Generate delta
                if get_serving().incremental_streaming_output:
                    delta = text
                else:
                    delta = text[offset:]
                stream_offsets[index] = len(content["text"])
                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Abort with an explicit error status_code is a system error
                # (timeout, OOM, validation): emit a streaming error chunk.
                # A graceful abort (no status_code, e.g. user-initiated via
                # /abort_request or session lifecycle cleanup) falls through
                # to the normal chunk path, matching the non-stream behavior
                # in tokenizer_manager._handle_abort_finish_reason.
                if finish_reason_type == "abort":
                    # Exempt from padding, including the graceful variant that
                    # falls through below: an abort is not content-correlated, and
                    # padding it would relabel a cancelled request as a normal
                    # length-capped completion and hide the cancellation.
                    aborted = True
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
                    break

                if padding is None or aborted:
                    choice_data = CompletionResponseStreamChoice(
                        index=index,
                        text=delta,
                        logprobs=logprobs,
                        finish_reason=finish_reason_type,
                        matched_stop=(
                            finish_reason["matched"]
                            if finish_reason and "matched" in finish_reason
                            else None
                        ),
                        token_ids=chunk_token_ids,
                        prompt_token_ids=chunk_prompt_token_ids,
                    )
                    chunk = CompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=created,
                        object="text_completion",
                        choices=[choice_data],
                        model=request.model,
                    )

                    # Add usage stats if continuous_usage_stats is enabled
                    if continuous_usage_stats:
                        chunk.usage = UsageProcessor.calculate_token_usage(
                            prompt_tokens=prompt_tokens.get(index, 0),
                            completion_tokens=completion_tokens.get(index, 0),
                            reasoning_tokens=reasoning_tokens.get(index, 0),
                        )

                    yield f"data: {chunk.model_dump_json()}\n\n"
                    stream_started = True
                    continue

                # Register the choice even on a chunk carrying no new token, so a
                # completion that emits nothing still gets a full pad run.
                wire_tokens.setdefault(index, 0)
                emitted = wire_tokens[index]
                new_tokens = self._chunk_token_count(content, emitted)

                group_text = pending_text.pop(index, "") + delta
                carry_prompt_ids = pending_prompt_ids.pop(index, None)
                if chunk_prompt_token_ids is not None:
                    carry_prompt_ids = chunk_prompt_token_ids
                if new_tokens == 0:
                    if group_text:
                        pending_text[index] = group_text
                    if carry_prompt_ids is not None:
                        pending_prompt_ids[index] = carry_prompt_ids
                    continue

                position_logprobs = (
                    padding.per_position_logprobs(logprobs, new_tokens)
                    if padding.return_logprob
                    else [None] * new_tokens
                )
                # One frame per token; the manager merges decode steps into a single
                # delta when the consumer lags, and an unsplit delta would leave the
                # frame count tracking the real length. Terminal frame comes with
                # the pad run.
                for offset_in_chunk in range(new_tokens):
                    wire_tokens[index] = emitted + offset_in_chunk + 1
                    yield self._stream_frame(
                        request=request,
                        response_id=content["meta_info"]["id"],
                        created=created,
                        choice=CompletionResponseStreamChoice(
                            index=index,
                            # The delta rides the group's last frame; slicing
                            # detokenized text per id would not reassemble
                            # byte-exactly for multi-token characters.
                            text=(
                                group_text if offset_in_chunk == new_tokens - 1 else ""
                            ),
                            logprobs=position_logprobs[offset_in_chunk],
                            finish_reason=None,
                            matched_stop=None,
                            token_ids=(
                                [chunk_token_ids[offset_in_chunk]]
                                if chunk_token_ids is not None
                                and offset_in_chunk < len(chunk_token_ids)
                                else None
                            ),
                            prompt_token_ids=(
                                carry_prompt_ids if offset_in_chunk == 0 else None
                            ),
                        ),
                        usage=(
                            UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                completion_tokens=wire_tokens[index],
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                            )
                            if continuous_usage_stats
                            else None
                        ),
                    )
                    stream_started = True

            if padding is not None and wire_tokens and not aborted:
                for index in sorted(wire_tokens):
                    for pad_frame in self._pad_run_frames(
                        request=request,
                        response_id=content["meta_info"]["id"],
                        created=created,
                        index=index,
                        padding=padding,
                        wire_tokens=wire_tokens,
                        manager_tokens=completion_tokens.get(index, 0),
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        continuous_usage_stats=continuous_usage_stats,
                        carry_text=pending_text.pop(index, ""),
                        carry_prompt_token_ids=pending_prompt_ids.pop(index, None),
                    ):
                        stream_started = True
                        yield pad_frame

            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        response_hidden_states = process_hidden_states_for_response(
                            choice_hidden_states, request.return_hidden_states
                        )
                        hidden_states_chunk = CompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=created,
                            object="text_completion",
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=index,
                                    text="",
                                    hidden_states=response_hidden_states,
                                    finish_reason=None,
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

            if any(
                obj is not None
                for obj in [
                    sglext_routed,
                    sglext_cached_tokens_details,
                    sglext_spec_tokens_details,
                ]
            ):
                sglext_chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    object="text_completion",
                    choices=[],  # sglext is at response level
                    model=request.model,
                    sglext=SglExt(
                        routed_experts=sglext_routed,
                        cached_tokens_details=sglext_cached_tokens_details,
                        spec_tokens_details=sglext_spec_tokens_details,
                    ),
                )
                yield f"data: {sglext_chunk.model_dump_json()}\n\n"

            # Handle final usage chunk
            if include_usage:
                # An honest count hands back the exact completion length the
                # padding exists to hide, so report what was framed instead.
                reported_completion_tokens = (
                    wire_tokens
                    if padding is not None and not aborted
                    else completion_tokens
                )
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    reported_completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=get_serving().enable_cache_report,
                )
                final_usage_chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    choices=[],
                    model=request.model,
                    usage=usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(exclude_none=True)
                yield f"data: {final_usage_data}\n\n"

        except Exception as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[CompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming completion request"""
        try:
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )
            ret = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_completion_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_completion_response(
        self,
        request: CompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> CompletionResponse:
        """Build completion response from generation results"""
        choices = []
        echo = False

        # Prepare echo prompts if needed
        echo_prompts = []
        if request.echo:
            echo_prompts = self._prepare_echo_prompts(request)
            echo = True

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
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
        response_sglext = None
        if routed_experts or cached_tokens_details or spec_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
                spec_tokens_details=spec_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            text = ret_item["text"]

            # Handle echo
            if echo:
                prompt_index = idx // request.n
                text = echo_prompts[prompt_index] + text

            # Handle logprobs
            logprobs = None
            if request.logprobs is not None:
                if echo:
                    input_token_logprobs = ret_item["meta_info"]["input_token_logprobs"]
                    input_top_logprobs = ret_item["meta_info"]["input_top_logprobs"]
                else:
                    input_token_logprobs = None
                    input_top_logprobs = None

                logprobs = to_openai_style_logprobs(
                    input_token_logprobs=input_token_logprobs,
                    input_top_logprobs=input_top_logprobs,
                    output_token_logprobs=ret_item["meta_info"].get(
                        "output_token_logprobs", []
                    ),
                    output_top_logprobs=ret_item["meta_info"].get(
                        "output_top_logprobs", []
                    ),
                )

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]

            choice_data = CompletionResponseChoice(
                index=idx,
                text=text,
                logprobs=logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
                token_ids=(
                    ret_item["output_ids"] if request.return_token_ids else None
                ),
                prompt_token_ids=(
                    ret_item.get("prompt_token_ids")
                    if request.return_token_ids
                    else None
                ),
            )
            choices.append(choice_data)

        # Calculate usage
        cache_report = get_serving().enable_cache_report
        usage = UsageProcessor.calculate_response_usage(
            ret, n_choices=request.n, enable_cache_report=cache_report
        )

        return CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            created=created,
            choices=choices,
            usage=usage,
            metadata=build_endpoint_weight_version_metadata(ret[0]["meta_info"]),
            sglext=response_sglext,
        )

    def _get_echo_text(self, request: CompletionRequest, index: int) -> str:
        """Get echo text for streaming response"""
        if isinstance(request.prompt, str):
            # for the case of single str prompts
            return request.prompt
        elif isinstance(request.prompt, list):
            if isinstance(request.prompt[0], str):
                # for the case of multiple str prompts
                return request.prompt[index // request.n]
            elif isinstance(request.prompt[0], int):
                # for the case of single token ids prompt
                return self.tokenizer_manager.tokenizer.decode(
                    request.prompt, skip_special_tokens=True
                )
            elif isinstance(request.prompt[0], list) and isinstance(
                request.prompt[0][0], int
            ):
                # for the case of multiple token ids prompts
                return self.tokenizer_manager.tokenizer.decode(
                    request.prompt[index // request.n],
                    skip_special_tokens=True,
                )
        return ""

    def _prepare_echo_prompts(self, request: CompletionRequest) -> List[str]:
        """Prepare echo prompts for non-streaming response"""
        # TODO: handle the case prompt is token ids
        if isinstance(request.prompt, list) and isinstance(request.prompt[0], str):
            # for the case of multiple str prompts
            return request.prompt
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], list):
            # for the case of multiple token ids prompts
            return [
                self.tokenizer_manager.tokenizer.decode(
                    prompt, skip_special_tokens=True
                )
                for prompt in request.prompt
            ]
        elif isinstance(request.prompt, list) and isinstance(request.prompt[0], int):
            # for the case of single token ids prompt
            return [
                self.tokenizer_manager.tokenizer.decode(
                    request.prompt, skip_special_tokens=True
                )
            ]
        else:
            # for the case of single str prompt
            return [request.prompt]
