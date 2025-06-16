import time
from typing import Any, Dict, List, Optional, Union

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
from sglang.srt.entrypoints.openai.utils import (
    aggregate_token_usage,
    to_openai_style_logprobs,
)
from sglang.srt.managers.io_struct import GenerateReqInput


class OpenAIServingCompletion(OpenAIServingBase):
    """Handler for completion requests"""

    def _request_id_prefix(self) -> str:
        return "cmpl-"

    def _validate_request(self, request: CompletionRequest) -> Optional[str]:
        """Validate completion prompt format and content"""
        if not (prompt := request.prompt):
            return "Prompt cannot be None"

        if isinstance(prompt, str):
            if not prompt.strip():
                return "Prompt cannot be empty or whitespace only"
        elif isinstance(prompt, list):
            if not prompt:
                return "Prompt list cannot be empty"

            # Check if it's a list of strings
            if all(isinstance(item, str) for item in prompt):
                for i, item in enumerate(prompt):
                    if not item.strip():
                        return f"Prompt at index {i} cannot be empty or whitespace only"

            # Check if it's a list of token IDs (integers)
            elif all(isinstance(item, int) for item in prompt):
                if any(item < 0 for item in prompt):
                    return "Token IDs must be non-negative"

            # Check if it's a list of lists (multiple token sequences)
            elif all(isinstance(item, list) for item in prompt):
                for i, item in enumerate(prompt):
                    if not item:
                        return f"Token sequence at index {i} cannot be empty"
                    if not all(isinstance(token, int) for token in item):
                        return f"Token sequence at index {i} must contain only integers"
                    if any(token < 0 for token in item):
                        return (
                            f"Token sequence at index {i} contains negative token IDs"
                        )
            else:
                return "Prompt must be string, list of strings, list of integers, or list of integer lists"
        else:
            return "Prompt must be string or list"

        return None

    def _convert_to_internal_request(
        self,
        all_requests: List[CompletionRequest],
        request_ids: List[str],
    ) -> tuple[GenerateReqInput, Union[CompletionRequest, List[CompletionRequest]]]:
        """Convert OpenAI completion request to internal format"""
        # Validate batch requests
        if len(all_requests) > 1:
            first_prompt_type = type(all_requests[0].prompt)
            for request in all_requests:
                assert (
                    type(request.prompt) is first_prompt_type
                ), "All prompts must be of the same type in file input settings"
                if request.n > 1:
                    raise ValueError(
                        "Parallel sampling is not supported for completions from files"
                    )

        prompts = []
        sampling_params_list = []
        return_logprobs = []
        logprob_start_lens = []
        top_logprobs_nums = []
        lora_paths = []

        for request in all_requests:
            # Process prompt
            prompt = request.prompt
            if is_completion_template_defined():
                prompt = generate_completion_prompt_from_request(request)

            prompts.append(prompt)

            lora_paths.append(request.lora_path)

            # Set logprob start length based on echo and logprobs
            if request.echo and request.logprobs:
                current_logprob_start_len = 0
            else:
                current_logprob_start_len = -1

            # Build sampling parameters
            sampling_params = self._build_sampling_params(request)
            sampling_params_list.append(sampling_params)

            return_logprobs.append(request.logprobs is not None)
            logprob_start_lens.append(current_logprob_start_len)
            top_logprobs_nums.append(
                request.logprobs if request.logprobs is not None else 0
            )

        # Handle single vs multiple requests
        if len(all_requests) == 1:
            if isinstance(prompts[0], str) or isinstance(prompts[0][0], str):
                prompt_kwargs = {"text": prompts[0]}
            else:
                prompt_kwargs = {"input_ids": prompts[0]}
            sampling_params_list = sampling_params_list[0]
            return_logprobs = return_logprobs[0]
            logprob_start_lens = logprob_start_lens[0]
            top_logprobs_nums = top_logprobs_nums[0]
            lora_paths = lora_paths[0]
            request_ids = request_ids[0]
        else:
            if isinstance(prompts[0], str) or isinstance(prompts[0][0], str):
                prompt_kwargs = {"text": prompts}
            else:
                prompt_kwargs = {"input_ids": prompts}

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            sampling_params=sampling_params_list,
            return_logprob=return_logprobs,
            top_logprobs_num=top_logprobs_nums,
            logprob_start_len=logprob_start_lens,
            return_text_in_logprobs=True,
            stream=all_requests[0].stream,
            rid=request_ids,
            lora_path=lora_paths,
            bootstrap_host=all_requests[0].bootstrap_host,
            bootstrap_port=all_requests[0].bootstrap_port,
            bootstrap_room=all_requests[0].bootstrap_room,
        )

        return adapted_request, (
            all_requests if len(all_requests) > 1 else all_requests[0]
        )

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

        # No additional completion-specific parameters needed currently
        # (json_schema is already handled in base method)

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming completion request"""
        created = int(time.time())

        async def generate_stream_resp():
            stream_buffers = {}
            n_prev_tokens = {}
            prompt_tokens = {}
            completion_tokens = {}
            cached_tokens = {}

            try:
                async for content in self.tokenizer_manager.generate_request(
                    adapted_request, raw_request
                ):
                    index = content.get("index", 0)

                    stream_buffer = stream_buffers.get(index, "")
                    n_prev_token = n_prev_tokens.get(index, 0)

                    text = content["text"]
                    prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                    completion_tokens[index] = content["meta_info"]["completion_tokens"]
                    cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)

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
                            input_top_logprobs = content["meta_info"][
                                "input_top_logprobs"
                            ]
                        else:
                            input_token_logprobs = None
                            input_top_logprobs = None

                        logprobs = to_openai_style_logprobs(
                            input_token_logprobs=input_token_logprobs,
                            input_top_logprobs=input_top_logprobs,
                            output_token_logprobs=content["meta_info"][
                                "output_token_logprobs"
                            ][n_prev_token:],
                            output_top_logprobs=content["meta_info"][
                                "output_top_logprobs"
                            ][n_prev_token:],
                        )
                        n_prev_token = len(
                            content["meta_info"]["output_token_logprobs"]
                        )

                    # Generate delta
                    delta = text[len(stream_buffer) :]
                    stream_buffer = stream_buffer + delta
                    finish_reason = content["meta_info"]["finish_reason"]

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

                    stream_buffers[index] = stream_buffer
                    n_prev_tokens[index] = n_prev_token

                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle final usage chunk
                if request.stream_options and request.stream_options.include_usage:
                    usage = self._calculate_streaming_usage_base(
                        prompt_tokens, completion_tokens, cached_tokens, request.n
                    )
                    final_usage_chunk = CompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=created,
                        choices=[],
                        model=request.model,
                        usage=usage,
                    )
                    final_usage_data = final_usage_chunk.model_dump_json(
                        exclude_none=True
                    )
                    yield f"data: {final_usage_data}\n\n"

            except Exception as e:
                error = self.create_streaming_error_response(str(e))
                yield f"data: {error}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream_resp(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

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
            cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return response

    def _build_completion_response(
        self,
        request: CompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
        cache_report: bool = False,
    ) -> CompletionResponse:
        """Build completion response from generation results"""
        choices = []
        echo = False

        # Prepare echo prompts if needed
        echo_prompts = []
        if (not isinstance(request, list)) and request.echo:
            echo_prompts = self._prepare_echo_prompts(request)
            echo = True

        for idx, ret_item in enumerate(ret):
            text = ret_item["text"]

            # Handle echo
            if isinstance(request, list) and request[idx].echo:
                echo = True
                text = request[idx].prompt + text
            elif echo and not isinstance(request, list):
                prompt_index = idx // request.n
                text = echo_prompts[prompt_index] + text

            # Handle logprobs
            logprobs = None
            if isinstance(request, list) and request[idx].logprobs is not None:
                logprobs = True
            elif (not isinstance(request, list)) and request.logprobs is not None:
                logprobs = True

            if logprobs:
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
            )
            choices.append(choice_data)

        # Calculate usage
        usage = aggregate_token_usage(ret, request.n, cache_report)

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
