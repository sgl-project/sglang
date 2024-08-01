"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Conversion between OpenAI APIs and native SRT APIs"""

import asyncio
import json
import os
import time
import uuid
from http import HTTPStatus
from typing import Dict, List, Optional

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.openai_api.protocol import (
    BatchRequest,
    BatchResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    FileRequest,
    FileResponse,
    LogProbs,
    TopLogprob,
    UsageInfo,
)

chat_template_name = None


class FileMetadata:
    def __init__(self, filename: str, purpose: str):
        self.filename = filename
        self.purpose = purpose


# In-memory storage for batch jobs and files
batch_storage: Dict[str, BatchResponse] = {}
file_id_request: Dict[str, FileMetadata] = {}
file_id_response: Dict[str, FileResponse] = {}
# map file id to file path in SGlang backend
file_id_storage: Dict[str, str] = {}


# backend storage directory
storage_dir = None


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


async def v1_files_create(file: UploadFile, purpose: str, file_storage_pth: str = None):
    try:
        global storage_dir
        if file_storage_pth:
            storage_dir = file_storage_pth
        # Read the file content
        file_content = await file.read()

        # Create an instance of RequestBody
        request_body = FileRequest(file=file_content, purpose=purpose)

        # Save the file to the sglang_oai_storage directory
        os.makedirs(storage_dir, exist_ok=True)
        file_id = f"backend_input_file-{uuid.uuid4()}"
        filename = f"{file_id}.jsonl"
        file_path = os.path.join(storage_dir, filename)

        with open(file_path, "wb") as f:
            f.write(request_body.file)

        # add info to global file map
        file_id_request[file_id] = FileMetadata(filename=file.filename, purpose=purpose)
        file_id_storage[file_id] = file_path

        # Return the response in the required format
        response = FileResponse(
            id=file_id,
            bytes=len(request_body.file),
            created_at=int(time.time()),
            filename=file.filename,
            purpose=request_body.purpose,
        )
        file_id_response[file_id] = response

        return response
    except ValidationError as e:
        return {"error": "Invalid input", "details": e.errors()}


async def v1_batches(tokenizer_manager, raw_request: Request):
    try:
        body = await raw_request.json()

        batch_request = BatchRequest(**body)

        batch_id = f"batch_{uuid.uuid4()}"

        # Create an instance of BatchResponse
        batch_response = BatchResponse(
            id=batch_id,
            endpoint=batch_request.endpoint,
            input_file_id=batch_request.input_file_id,
            completion_window=batch_request.completion_window,
            created_at=int(time.time()),
            metadata=batch_request.metadata,
        )

        batch_storage[batch_id] = batch_response

        # Start processing the batch asynchronously
        asyncio.create_task(process_batch(tokenizer_manager, batch_id, batch_request))

        # Return the initial batch_response
        return batch_response

    except ValidationError as e:
        return {"error": "Invalid input", "details": e.errors()}
    except Exception as e:
        return {"error": str(e)}


async def process_batch(tokenizer_manager, batch_id: str, batch_request: BatchRequest):
    try:
        # Update the batch status to "in_progress"
        batch_storage[batch_id].status = "in_progress"
        batch_storage[batch_id].in_progress_at = int(time.time())

        # Retrieve the input file content
        input_file_request = file_id_request.get(batch_request.input_file_id)
        if not input_file_request:
            raise ValueError("Input file not found")

        # Parse the JSONL file and process each request
        input_file_path = file_id_storage.get(batch_request.input_file_id)
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_requests = len(lines)
        completed_requests = 0
        failed_requests = 0

        all_ret = []
        end_point = batch_storage[batch_id].endpoint
        file_request_list = []
        all_requests = []
        for line in lines:
            request_data = json.loads(line)
            file_request_list.append(request_data)
            body = request_data["body"]
            if end_point == "/v1/chat/completions":
                all_requests.append(ChatCompletionRequest(**body))
            elif end_point == "/v1/completions":
                all_requests.append(CompletionRequest(**body))
        if end_point == "/v1/chat/completions":
            adapted_request, request = v1_chat_generate_request(
                all_requests, tokenizer_manager
            )
        elif end_point == "/v1/completions":
            adapted_request, request = v1_generate_request(all_requests)
        try:
            ret = await tokenizer_manager.generate_request(adapted_request).__anext__()
            if not isinstance(ret, list):
                ret = [ret]
            if end_point == "/v1/chat/completions":
                responses = v1_chat_generate_response(request, ret, to_file=True)
            else:
                responses = v1_generate_response(request, ret, to_file=True)

        except Exception as e:
            error_json = {
                "id": f"batch_req_{uuid.uuid4()}",
                "custom_id": request_data.get("custom_id"),
                "response": None,
                "error": {"message": str(e)},
            }
            all_ret.append(error_json)
            failed_requests += len(file_request_list)

        for idx, response in enumerate(responses):
            # the batch_req here can be changed to be named within a batch granularity
            response_json = {
                "id": f"batch_req_{uuid.uuid4()}",
                "custom_id": file_request_list[idx].get("custom_id"),
                "response": response,
                "error": None,
            }
            all_ret.append(response_json)
            completed_requests += 1
        # Write results to a new file
        output_file_id = f"backend_result_file-{uuid.uuid4()}"
        global storage_dir
        output_file_path = os.path.join(storage_dir, f"{output_file_id}.jsonl")
        with open(output_file_path, "w", encoding="utf-8") as f:
            for ret in all_ret:
                f.write(json.dumps(ret) + "\n")

        # Update batch response with output file information
        retrieve_batch = batch_storage[batch_id]
        retrieve_batch.output_file_id = output_file_id
        file_id_storage[output_file_id] = output_file_path
        # Update batch status to "completed"
        retrieve_batch.status = "completed"
        retrieve_batch.completed_at = int(time.time())
        retrieve_batch.request_counts = {
            "total": total_requests,
            "completed": completed_requests,
            "failed": failed_requests,
        }

    except Exception as e:
        print("error in SGlang:", e)
        # Update batch status to "failed"
        retrieve_batch = batch_storage[batch_id]
        retrieve_batch.status = "failed"
        retrieve_batch.failed_at = int(time.time())
        retrieve_batch.errors = {"message": str(e)}


async def v1_retrieve_batch(batch_id: str):
    # Retrieve the batch job from the in-memory storage
    batch_response = batch_storage.get(batch_id)
    if batch_response is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    return batch_response


async def v1_retrieve_file(file_id: str):
    # Retrieve the batch job from the in-memory storage
    file_response = file_id_response.get(file_id)
    if file_response is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file_response


async def v1_retrieve_file_content(file_id: str):
    file_pth = file_id_storage.get(file_id)
    if not file_pth or not os.path.exists(file_pth):
        raise HTTPException(status_code=404, detail="File not found")

    def iter_file():
        with open(file_pth, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iter_file(), media_type="application/octet-stream")


def v1_generate_request(all_requests):

    prompts = []
    sampling_params_list = []
    return_logprobs = []
    top_logprobs_nums = []
    first_prompt_type = type(all_requests[0].prompt)
    for request in all_requests:
        prompt = request.prompt
        assert (
            type(prompt) == first_prompt_type
        ), "All prompts must be of the same type in file input settings"
        prompts.append(prompt)
        return_logprobs.append(request.logprobs is not None and request.logprobs > 0)
        top_logprobs_nums.append(
            request.logprobs if request.logprobs is not None else 0
        )
        sampling_params_list.append(
            {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "stop": request.stop,
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "regex": request.regex,
                "n": request.n,
                "ignore_eos": request.ignore_eos,
            }
        )
        if len(all_requests) > 1 and request.n > 1:
            raise ValueError(
                "Batch operation is not supported for completions from files"
            )

    if len(all_requests) == 1:
        prompt = prompts[0]
        sampling_params_list = sampling_params_list[0]
        return_logprobs = return_logprobs[0]
        top_logprobs_nums = top_logprobs_nums[0]
        if isinstance(prompt, str) or isinstance(prompt[0], str):
            prompt_kwargs = {"text": prompt}
        else:
            prompt_kwargs = {"input_ids": prompt}
    else:
        if isinstance(prompts[0], str):
            prompt_kwargs = {"text": prompts}
        else:
            prompt_kwargs = {"input_ids": prompts}
    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        sampling_params=sampling_params_list,
        return_logprob=return_logprobs,
        top_logprobs_num=top_logprobs_nums,
        return_text_in_logprobs=True,
        stream=all_requests[0].stream,
    )
    if len(all_requests) == 1:
        return adapted_request, all_requests[0]
    return adapted_request, all_requests


def v1_generate_response(request, ret, to_file=False):
    choices = []
    echo = False

    if (not isinstance(request, List)) and request.echo:
        # TODO: handle the case propmt is token ids
        if isinstance(request.prompt, list):
            prompts = request.prompt
        else:
            prompts = [request.prompt]
        echo = True

    for idx, ret_item in enumerate(ret):
        text = ret_item["text"]
        if isinstance(request, List) and request[idx].echo:
            echo = True
            text = request[idx].prompt + text
        if (not isinstance(request, List)) and echo:
            text = prompts[idx] + text

        logprobs = False
        if isinstance(request, List) and request[idx].logprobs:
            logprobs = True
        elif (not isinstance(request, List)) and request.logprobs:
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
                output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
                output_top_logprobs=ret_item["meta_info"]["output_top_logprobs"],
            )
        else:
            logprobs = None

        if to_file:
            # to make the choise data json serializable
            choice_data = {
                "index": 0,
                "text": text,
                "logprobs": logprobs,
                "finish_reason": ret_item["meta_info"]["finish_reason"],
            }
        else:
            choice_data = CompletionResponseChoice(
                index=idx,
                text=text,
                logprobs=logprobs,
                finish_reason=ret_item["meta_info"]["finish_reason"],
            )

        choices.append(choice_data)

    if to_file:
        responses = []
        for i, choice in enumerate(choices):
            response = {
                "status_code": 200,
                "request_id": ret[i]["meta_info"]["id"],
                "body": {
                    # remain the same but if needed we can change that
                    "id": ret[i]["meta_info"]["id"],
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request[i].model,
                    "choices": choice,
                    "usage": {
                        "prompt_tokens": ret[i]["meta_info"]["prompt_tokens"],
                        "completion_tokens": ret[i]["meta_info"]["completion_tokens"],
                        "total_tokens": ret[i]["meta_info"]["prompt_tokens"]
                        + ret[i]["meta_info"]["completion_tokens"],
                    },
                    "system_fingerprint": None,
                },
            }
            responses.append(response)
        return responses
    else:
        completion_tokens = sum(item["meta_info"]["completion_tokens"] for item in ret)
        response = CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=ret[0]["meta_info"]["prompt_tokens"],
                completion_tokens=completion_tokens,
                total_tokens=ret[0]["meta_info"]["prompt_tokens"] + completion_tokens,
            ),
        )
    return response


async def v1_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    all_requests = [CompletionRequest(**request_json)]
    adapted_request, request = v1_generate_request(all_requests)

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
                    else:
                        logprobs = None

                    delta = text[len(stream_buffer) :]
                    stream_buffer = stream_buffer + delta
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

    if not isinstance(ret, list):
        ret = [ret]

    response = v1_generate_response(request, ret)
    return response


def v1_chat_generate_request(all_requests, tokenizer_manager):

    texts = []
    sampling_params_list = []
    image_data_list = []
    return_logprobs = []
    top_logprobs_nums = []
    for request in all_requests:
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
        texts.append(prompt)
        return_logprobs.append(request.logprobs)
        top_logprobs_nums.append(request.top_logprobs)
        sampling_params_list.append(
            {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "stop": stop,
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "regex": request.regex,
                "n": request.n,
            }
        )
        image_data_list.append(image_data)
    if len(all_requests) == 1:
        texts = texts[0]
        sampling_params_list = sampling_params_list[0]
        image_data = image_data_list[0]
        return_logprobs = return_logprobs[0]
        top_logprobs_nums = top_logprobs_nums[0]
    adapted_request = GenerateReqInput(
        text=texts,
        image_data=image_data,
        sampling_params=sampling_params_list,
        return_logprob=return_logprobs,
        top_logprobs_num=top_logprobs_nums,
        stream=all_requests[0].stream,
        return_text_in_logprobs=True,
    )
    if len(all_requests) == 1:
        return adapted_request, all_requests[0]
    return adapted_request, all_requests


def v1_chat_generate_response(request, ret, to_file=False):
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, ret_item in enumerate(ret):
        logprobs = False
        if isinstance(request, List) and request[idx].logprobs:
            logprobs = True
        elif (not isinstance(request, List)) and request.logprobs:
            logprobs = True
        if logprobs:
            logprobs = to_openai_style_logprobs(
                output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
                output_top_logprobs=ret_item["meta_info"]["output_top_logprobs"],
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

            choice_logprobs = ChoiceLogprobs(content=token_logprobs)
        else:
            choice_logprobs = None
        prompt_tokens = ret_item["meta_info"]["prompt_tokens"]
        completion_tokens = ret_item["meta_info"]["completion_tokens"]

        if to_file:
            # to make the choice data json serializable
            choice_data = {
                "index": 0,
                "message": {"role": "assistant", "content": ret_item["text"]},
                "logprobs": choice_logprobs,
                "finish_reason": ret_item["meta_info"]["finish_reason"],
            }
        else:
            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(role="assistant", content=ret_item["text"]),
                logprobs=choice_logprobs,
                finish_reason=ret_item["meta_info"]["finish_reason"],
            )

        choices.append(choice_data)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
    if to_file:
        responses = []

        for i, choice in enumerate(choices):
            response = {
                "status_code": 200,
                "request_id": ret[i]["meta_info"]["id"],
                "body": {
                    # remain the same but if needed we can change that
                    "id": ret[i]["meta_info"]["id"],
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request[i].model,
                    "choices": choice,
                    "usage": {
                        "prompt_tokens": ret[i]["meta_info"]["prompt_tokens"],
                        "completion_tokens": ret[i]["meta_info"]["completion_tokens"],
                        "total_tokens": ret[i]["meta_info"]["prompt_tokens"]
                        + ret[i]["meta_info"]["completion_tokens"],
                    },
                    "system_fingerprint": None,
                },
            }
            responses.append(response)
        return responses
    else:
        response = ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )
        return response


async def v1_chat_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    all_requests = [ChatCompletionRequest(**request_json)]
    adapted_request, request = v1_chat_generate_request(all_requests, tokenizer_manager)

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
                    stream_buffer = stream_buffer + delta
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

    if not isinstance(ret, list):
        ret = [ret]

    response = v1_chat_generate_response(request, ret)

    return response


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
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

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs
