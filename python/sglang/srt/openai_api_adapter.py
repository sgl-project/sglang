"""Conversion between OpenAI APIs and native SRT APIs"""

import asyncio
import json
import os
from http import HTTPStatus

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    LogProbs,
    UsageInfo,
)

chat_template_name = None


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
):
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    return JSONResponse(content=error.model_dump(), status_code=error.code)


def create_streaming_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> str:
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    json_str = json.dumps({"error": error.model_dump()})
    return json_str


def load_chat_template_for_openai_api(chat_template_arg):
    global chat_template_name

    print(f"Use chat template: {chat_template_arg}")
    if not chat_template_exists(chat_template_arg):
        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )
        with open(chat_template_arg, "r") as filep:
            template = json.load(filep)
            try:
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None
            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        chat_template_name = template["name"]
    else:
        chat_template_name = chat_template_arg


async def v1_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    request = CompletionRequest(**request_json)

    if request.n != 1:
        return create_error_response("n != 1 is not supported")

    adapted_request = GenerateReqInput(
        text=request.prompt,
        sampling_params={
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "stop": request.stop,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "regex": request.regex,
        },
        return_logprob=request.logprobs is not None and request.logprobs > 0,
        top_logprobs_num=request.logprobs if request.logprobs is not None else 0,
        return_text_in_logprobs=True,
        stream=request.stream,
    )

    if adapted_request.stream:

        async def generate_stream_resp():
            stream_buffer = ""
            n_prev_token = 0
            try:
                async for content in tokenizer_manager.generate_request(
                    adapted_request, raw_request
                ):
                    text = content["text"]
                    prompt_tokens = content["meta_info"]["prompt_tokens"]
                    completion_tokens = content["meta_info"]["completion_tokens"]

                    if not stream_buffer:  # The first chunk
                        if request.echo:
                            # Prepend prompt in response text.
                            text = request.prompt + text

                    if request.logprobs:
                        # The first chunk and echo is enabled.
                        if not stream_buffer and request.echo:
                            prefill_token_logprobs = content["meta_info"][
                                "prefill_token_logprobs"
                            ]
                            prefill_top_logprobs = content["meta_info"][
                                "prefill_top_logprobs"
                            ]
                        else:
                            prefill_token_logprobs = None
                            prefill_top_logprobs = None

                        logprobs = to_openai_style_logprobs(
                            prefill_token_logprobs=prefill_token_logprobs,
                            prefill_top_logprobs=prefill_top_logprobs,
                            decode_token_logprobs=content["meta_info"][
                                "decode_token_logprobs"
                            ][n_prev_token:],
                            decode_top_logprobs=content["meta_info"][
                                "decode_top_logprobs"
                            ][n_prev_token:],
                        )

                        n_prev_token = len(
                            content["meta_info"]["decode_token_logprobs"]
                        )
                    else:
                        logprobs = None

                    delta = text[len(stream_buffer) :]
                    stream_buffer = content["text"]
                    choice_data = CompletionResponseStreamChoice(
                        index=0,
                        text=delta,
                        logprobs=logprobs,
                        finish_reason=content["meta_info"]["finish_reason"],
                    )
                    chunk = CompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        object="text_completion",
                        choices=[choice_data],
                        model=request.model,
                        usage=UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except ValueError as e:
                error = create_streaming_error_response(str(e))
                yield f"data: {error}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream_resp(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(adapted_request),
        )

    # Non-streaming response.
    try:
        ret = await tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
    except ValueError as e:
        return create_error_response(str(e))

    ret = ret[0] if isinstance(ret, list) else ret
    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
    text = ret["text"]
    if request.echo:
        text = request.prompt + text

    if request.logprobs:
        if request.echo:
            prefill_token_logprobs = ret["meta_info"]["prefill_token_logprobs"]
            prefill_top_logprobs = ret["meta_info"]["prefill_top_logprobs"]
        else:
            prefill_token_logprobs = None
            prefill_top_logprobs = None

        logprobs = to_openai_style_logprobs(
            prefill_token_logprobs=prefill_token_logprobs,
            prefill_top_logprobs=prefill_top_logprobs,
            decode_token_logprobs=ret["meta_info"]["decode_token_logprobs"],
            decode_top_logprobs=ret["meta_info"]["decode_top_logprobs"],
        )
    else:
        logprobs = None

    choice_data = CompletionResponseChoice(
        index=0,
        text=text,
        logprobs=logprobs,
        finish_reason=ret["meta_info"]["finish_reason"],
    )
    response = CompletionResponse(
        id=ret["meta_info"]["id"],
        model=request.model,
        choices=[choice_data],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


async def v1_chat_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    if request.n != 1:
        return create_error_response("n != 1 is not supported")

    # Prep the data needed for the underlying GenerateReqInput:
    #  - prompt: The full prompt string.
    #  - stop: Custom stop tokens.
    #  - image_data: None or a list of image strings (URLs or base64 strings).
    #    None skips any image processing in GenerateReqInput.
    if not isinstance(request.messages, str):
        # Apply chat template and its stop strings.
        if chat_template_name is None:
            prompt = tokenizer_manager.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            stop = request.stop
            image_data = None
        else:
            conv = generate_chat_conv(request, chat_template_name)
            prompt = conv.get_prompt()
            image_data = conv.image_data
            stop = conv.stop_str or []
            if request.stop:
                if isinstance(request.stop, str):
                    stop.append(request.stop)
                else:
                    stop.extend(request.stop)
    else:
        # Use the raw prompt and stop strings if the messages is already a string.
        prompt = request.messages
        stop = request.stop
        image_data = None

    adapted_request = GenerateReqInput(
        text=prompt,
        image_data=image_data,
        sampling_params={
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "stop": stop,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "regex": request.regex,
        },
        stream=request.stream,
    )

    if adapted_request.stream:

        async def generate_stream_resp():
            is_first = True

            stream_buffer = ""
            try:
                async for content in tokenizer_manager.generate_request(
                    adapted_request, raw_request
                ):
                    if is_first:
                        # First chunk with role
                        is_first = False
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant"),
                            finish_reason=content["meta_info"]["finish_reason"],
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    text = content["text"]
                    delta = text[len(stream_buffer) :]
                    stream_buffer = text
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=delta),
                        finish_reason=content["meta_info"]["finish_reason"],
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except ValueError as e:
                error = create_streaming_error_response(str(e))
                yield f"data: {error}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream_resp(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(adapted_request),
        )

    # Non-streaming response.
    try:
        ret = await tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
    except ValueError as e:
        return create_error_response(str(e))

    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=ret["text"]),
        finish_reason=ret["meta_info"]["finish_reason"],
    )
    response = ChatCompletionResponse(
        id=ret["meta_info"]["id"],
        model=request.model,
        choices=[choice_data],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


def to_openai_style_logprobs(
    prefill_token_logprobs=None,
    decode_token_logprobs=None,
    prefill_top_logprobs=None,
    decode_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if prefill_token_logprobs is not None:
        append_token_logprobs(prefill_token_logprobs)
    if decode_token_logprobs is not None:
        append_token_logprobs(decode_token_logprobs)
    if prefill_top_logprobs is not None:
        append_top_logprobs(prefill_top_logprobs)
    if decode_top_logprobs is not None:
        append_top_logprobs(decode_top_logprobs)

    return ret_logprobs
