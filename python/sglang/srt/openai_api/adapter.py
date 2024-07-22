"""Conversion between OpenAI APIs and native SRT APIs"""

import asyncio
import json
import os
import uuid
import time
from http import HTTPStatus

from fastapi import Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.openai_api.protocol import (
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
    FileRequest,
    FileResponse,
    BatchRequest,
    BatchResponse,
)

from pydantic import ValidationError
from typing import Optional, Dict

chat_template_name = None

# In-memory storage for batch jobs and files
batch_storage: Dict[str, BatchResponse] = {}
file_id_request: Dict[str, FileRequest] = {}
file_id_response: Dict[str, FileResponse] = {}
## map file id to file path in SGlang backend
file_id_storage: Dict[str, str] = {}


# backend storage directory
storage_dir = "/home/ubuntu/my_sglang_dev/sglang/python/sglang/srt/openai_api/sglang_oai_storage"




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

async def v1_files_create(file: UploadFile, purpose: str):
    try:
        # Read the file content
        file_content = await file.read()
        
        # Create an instance of RequestBody
        request_body = FileRequest(file=file_content, purpose=purpose)
        
        # Save the file to the sglang_oai_storage directory
        os.makedirs(storage_dir, exist_ok=True)
        file_id = f"backend_input_file-{uuid.uuid4()}"
        filename = f"{file_id}.jsonl"
        file_path = os.path.join(storage_dir, filename)
        
        print('file id in creat:', file_id)
        
        with open(file_path, "wb") as f:
            f.write(request_body.file)
        
        # add info to global file map
        file_id_request[file_id] = request_body
        file_id_storage[file_id] = file_path
        
        # Return the response in the required format
        response =  FileResponse(
                id=file_id,
                bytes=len(request_body.file),
                created_at=int(time.time()),
                filename=file.filename,
                purpose=request_body.purpose
                )
        file_id_response[file_id] = response
        
        return response
    except ValidationError as e:
        return {"error": "Invalid input", "details": e.errors()}
    

async def v1_batches(tokenizer_manager, raw_request: Request):
    try:
        # Parse the JSON body
        body = await raw_request.json()
        
        # Create an instance of BatchRequest
        batch_request = BatchRequest(**body)
        
        # Generate a unique batch ID
        batch_id = f"batch_{uuid.uuid4()}"
        
        # Create an instance of BatchResponse
        batch_response = BatchResponse(
            id=batch_id,
            endpoint=batch_request.endpoint,
            input_file_id=batch_request.input_file_id,
            completion_window=batch_request.completion_window,
            created_at=int(time.time()),
            metadata=batch_request.metadata
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
        print('batch_id in process_batch in SGlang backend:', batch_id)
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
        for line in lines:
            request_data = json.loads(line)
            if batch_storage[batch_id].endpoint == "/v1/chat/completions":
                adapted_request, request = v1_chat_generate_request(request_data, tokenizer_manager,from_file=True)
            elif batch_storage[batch_id].endpoint == "/v1/completions":
                pass
                

            try:
                
                print('adapted_request in SGlang in batch:', adapted_request)
                print('request_data in SGlang in batch:', request_data)
                
                ret = await tokenizer_manager.generate_request(adapted_request).__anext__()
                print('ret in SGlang:', ret)
                
                
                if not isinstance(ret, list):
                    ret = [ret]
                    
                response = v1_chat_generate_response(request, ret, to_file=True)
                response_json = {
                    "id": f"batch_req_{uuid.uuid4()}",
                    "custom_id": request_data.get("custom_id"),
                    "response": response,
                    "error": None
                }
                all_ret.append(response_json)
                
                completed_requests += 1
                print('success in SGlang:', ret)
            except Exception as e:
                error_json = {
                    "id": f"batch_req_{uuid.uuid4()}",
                    "custom_id": request_data.get("custom_id"),
                    "response": None,
                    "error": {"message": str(e)}
                }
                all_ret.append(error_json)
                failed_requests += 1
                continue
        print('all_ret in SGlang:', all_ret)
        
        
        # Write results to a new file
        output_file_id = f"backend_result_file-{uuid.uuid4()}"
        output_file_path = os.path.join(storage_dir, f"{output_file_id}.jsonl")
        print('output file id in SGlang:', output_file_id)
        with open(output_file_path, "w", encoding="utf-8") as f:
            for ret in all_ret:
                f.write(json.dumps(ret) + "\n")
        
        # Update batch response with output file information
        batch_storage[batch_id].output_file_id = output_file_id
        file_id_storage[output_file_id] = output_file_path
        # Update batch status to "completed"
        batch_storage[batch_id].status = "completed"
        batch_storage[batch_id].completed_at = int(time.time())
        batch_storage[batch_id].request_counts = {
            "total": total_requests,
            "completed": completed_requests,
            "failed": failed_requests
        }
        
    except Exception as e:
        print('error in SGlang:', e)
        # Update batch status to "failed"
        batch_storage[batch_id].status = "failed"
        batch_storage[batch_id].failed_at = int(time.time())
        batch_storage[batch_id].errors = {"message": str(e)}


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
    file_pth= file_id_storage.get(file_id)
    if not file_pth or not os.path.exists(file_pth):
        raise HTTPException(status_code=404, detail="File not found")

    def iter_file():
        with open(file_pth, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iter_file(), media_type="application/octet-stream")

async def v1_completions(tokenizer_manager, raw_request: Request):
    print('raw request in v1_completions of adapter.py', raw_request)
    request_json = await raw_request.json()
    print('in v1_completions of adapter.py')
    print(request_json)
    request = CompletionRequest(**request_json)

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
            "n": request.n,
            "ignore_eos": request.ignore_eos,
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
    choices = []

    for idx, ret_item in enumerate(ret):
        text = ret_item["text"]

        if request.echo:
            text = request.prompt + text

        if request.logprobs:
            if request.echo:
                prefill_token_logprobs = ret_item["meta_info"]["prefill_token_logprobs"]
                prefill_top_logprobs = ret_item["meta_info"]["prefill_top_logprobs"]
            else:
                prefill_token_logprobs = None
                prefill_top_logprobs = None

            logprobs = to_openai_style_logprobs(
                prefill_token_logprobs=prefill_token_logprobs,
                prefill_top_logprobs=prefill_top_logprobs,
                decode_token_logprobs=ret_item["meta_info"]["decode_token_logprobs"],
                decode_top_logprobs=ret_item["meta_info"]["decode_top_logprobs"],
            )
        else:
            logprobs = None

        choice_data = CompletionResponseChoice(
            index=idx,
            text=text,
            logprobs=logprobs,
            finish_reason=ret_item["meta_info"]["finish_reason"],
        )

        choices.append(choice_data)

    response = CompletionResponse(
        id=ret[0]["meta_info"]["id"],
        model=request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=ret[0]["meta_info"]["prompt_tokens"],
            completion_tokens=sum(
                item["meta_info"]["completion_tokens"] for item in ret
            ),
            total_tokens=ret[0]["meta_info"]["prompt_tokens"]
            + sum(item["meta_info"]["completion_tokens"] for item in ret),
        ),
    )

    return response

def v1_chat_generate_request(request_json, tokenizer_manager, from_file=False):
    if from_file:
        body = request_json["body"]
        request_data = {
            "messages": body["messages"],
            "model": body["model"],
            "frequency_penalty": body.get("frequency_penalty", 0.0),
            "logit_bias": body.get("logit_bias", None),
            "logprobs": body.get("logprobs", False),
            "top_logprobs": body.get("top_logprobs", None),
            "max_tokens": body.get("max_tokens", 16),
            "n": body.get("n", 1),
            "presence_penalty": body.get("presence_penalty", 0.0),
            "response_format": body.get("response_format", None),
            "seed": body.get("seed", None),
            "stop": body.get("stop", []),
            "stream": body.get("stream", False),
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 1.0),
            "user": body.get("user", None),
            "regex": body.get("regex", None)
        }
        request = ChatCompletionRequest(**request_data)
        ## TODO collect custom id for reorder
    else:
        request = ChatCompletionRequest(**request_json)
    
    print('request messages in v1_chat_completions:', request.messages)
    
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
            "n": request.n,
        },
        stream=request.stream,
    )
    return adapted_request, request


def v1_chat_generate_response(request, ret, to_file=False):
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, ret_item in enumerate(ret):
        prompt_tokens = ret_item["meta_info"]["prompt_tokens"]
        completion_tokens = ret_item["meta_info"]["completion_tokens"]

        if to_file:
            choice_data = {
                "index": idx,
                "message": {"role": "assistant", "content": ret_item["text"]},
                "logprobs": None,
                "finish_reason": ret_item["meta_info"]["finish_reason"],
            }
        else:
            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(role="assistant", content=ret_item["text"]),
                finish_reason=ret_item["meta_info"]["finish_reason"],
            )

        choices.append(choice_data)
        total_prompt_tokens = prompt_tokens
        total_completion_tokens += completion_tokens

    if to_file:
        response = {
            "status_code": 200,
            "request_id": ret[0]["meta_info"]["id"],
            "body": {
                "id": ret[0]["meta_info"]["id"],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                },
                "system_fingerprint": None
            }
        }
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
    
    print('request json in v1_chat_completions:', request_json)
    
    adapted_request, request = v1_chat_generate_request(request_json, tokenizer_manager, from_file=False)
    
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
        print('adapted_request in v1_chat_completions:', adapted_request)
        print('raw_request in v1_chat_completions:', raw_request)
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
