import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Union

from fastapi import Request
from fastapi.responses import StreamingResponse

from sglang.srt.code_completion_parser import (
    generate_completion_prompt_from_request,
    is_completion_template_defined,
)
from sglang.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_hidden_states_from_ret,
    to_openai_style_logprobs,
)
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


class OpenAIServingCompletion(OpenAIServingBase):
    """Handler for completion requests"""

    def _request_id_prefix(self) -> str:
        return "cmpl-"

    def _convert_to_internal_request(
        self,
        request: CompletionRequest,
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
        if is_completion_template_defined():
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

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            sampling_params=sampling_params,
            return_logprob=request.logprobs is not None,
            top_logprobs_num=request.logprobs if request.logprobs is not None else 0,
            logprob_start_len=logprob_start_len,
            return_text_in_logprobs=True,
            stream=request.stream,
            lora_path=request.lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            return_hidden_states=request.return_hidden_states,
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
        }

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming completion request"""
        return StreamingResponse(
            self._generate_completion_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
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
        stream_buffers = {}
        n_prev_tokens = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        cached_tokens = {}

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                text = content["text"]
                prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                completion_tokens[index] = content["meta_info"]["completion_tokens"]
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)

                stream_buffer = stream_buffers.get(index, "")
                # Handle echo for first chunk
                if not stream_buffer:  # The first chunk
                    if request.echo:
                        echo_text = self._get_echo_text(request, index)
                        text = echo_text + text

                # Handle logprobs
                logprobs = None
                if request.logprobs is not None:
                    # The first chunk and echo is enabled.
                    if not stream_buffer and request.echo:
                        input_token_logprobs = content["meta_info"][
                            "input_token_logprobs"
                        ]
                        input_top_logprobs = content["meta_info"]["input_top_logprobs"]
                    else:
                        input_token_logprobs = None
                        input_top_logprobs = None

                    n_prev_token = n_prev_tokens.get(index, 0)
                    logprobs = to_openai_style_logprobs(
                        input_token_logprobs=input_token_logprobs,
                        input_top_logprobs=input_top_logprobs,
                        output_token_logprobs=content["meta_info"][
                            "output_token_logprobs"
                        ][n_prev_token:],
                        output_top_logprobs=content["meta_info"]["output_top_logprobs"][
                            n_prev_token:
                        ],
                    )
                    n_prev_tokens[index] = len(
                        content["meta_info"]["output_token_logprobs"]
                    )

                # Generate delta
                delta = text[len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta
                finish_reason = content["meta_info"]["finish_reason"]
                hidden_states = content["meta_info"].get("hidden_states", None)

                choice_data = CompletionResponseStreamChoice(
                    index=index,
                    text=delta,
                    logprobs=logprobs,
                    finish_reason=finish_reason["type"] if finish_reason else None,
                    matched_stop=(
                        finish_reason["matched"]
                        if finish_reason and "matched" in finish_reason
                        else None
                    ),
                )
                chunk = CompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=created,
                    object="text_completion",
                    choices=[choice_data],
                    model=request.model,
                )

                yield f"data: {chunk.model_dump_json()}\n\n"

            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = CompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=created,
                            object="text_completion",
                            choices=[
                                CompletionResponseStreamChoice(
                                    index=index,
                                    text="",
                                    hidden_states=last_token_hidden_states,
                                    finish_reason=None,
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            # Handle final usage chunk
            if request.stream_options and request.stream_options.include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
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
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[CompletionResponse, ErrorResponse]:
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
                    output_token_logprobs=ret_item["meta_info"][
                        "output_token_logprobs"
                    ],
                    output_top_logprobs=ret_item["meta_info"]["output_top_logprobs"],
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
            )
            choices.append(choice_data)

        # Calculate usage
        cache_report = self.tokenizer_manager.server_args.enable_cache_report
        usage = UsageProcessor.calculate_response_usage(
            ret, n_choices=request.n, enable_cache_report=cache_report
        )

        return CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            created=created,
            choices=choices,
            usage=usage,
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
