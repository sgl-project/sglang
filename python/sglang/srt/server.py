"""SRT: SGLang Runtime"""

import asyncio
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import List, Optional, Union

# Fix a Python bug
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.constrained import disable_cache
from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import DetokenizeReqInput, GenerateReqInput
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
    LogProbs,
    UsageInfo,
)
from sglang.srt.managers.router.manager import start_router_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    enable_show_time_cost,
    allocate_init_ports,
    jsonify_pydantic_model,
    assert_pkg_version,
    get_exception_traceback,
    API_KEY_HEADER_NAME,
    APIKeyValidatorMiddleware
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None
chat_template_name = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
    }
    return result


@app.get("/get_server_args")
async def get_server_args():
    return dataclasses.asdict(tokenizer_manager.server_args)


@app.get("/flush_cache")
async def flush_cache():
    await tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def stream_generator(obj: GenerateReqInput):
    async for out in tokenizer_manager.generate_request(obj):
        await handle_token_logprobs_results(obj, out)
        yield out


@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
    obj.post_init()

    if obj.stream:

        async def stream_results():
            async for out in stream_generator(obj):
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    ret = await tokenizer_manager.generate_request(obj).__anext__()
    await handle_token_logprobs_results(obj, ret)

    return ret


async def detokenize_logprob_tokens(token_logprobs, decode_to_text):
    if not decode_to_text:
        return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

    token_ids = [tid for _, tid in token_logprobs]
    token_texts = await tokenizer_manager.detokenize(DetokenizeReqInput(token_ids))
    return [
        (logprob, token_id, token_text)
        for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
    ]


async def detokenize_top_logprobs_tokens(top_logprobs, decode_to_text):
    for i, t in enumerate(top_logprobs):
        if top_logprobs[i] is not None:
            top_logprobs[i] = await detokenize_logprob_tokens(t, decode_to_text)
    return top_logprobs


async def handle_token_logprobs_results(obj: GenerateReqInput, ret):
    """Handle the token logprobs results, convert token ids to text if needed.

    Args:
        obj (GenerateReqInput): The request object.
        ret (Union[Dict, List[Dict]]): The response object.
    """
    # NOTE: This is because the multiple requests in one http request.

    async def convert_style(r, return_text):
        r["meta_info"]["prefill_token_logprobs"] = await detokenize_logprob_tokens(
            r["meta_info"]["prefill_token_logprobs"], return_text
        )
        r["meta_info"]["decode_token_logprobs"] = await detokenize_logprob_tokens(
            r["meta_info"]["decode_token_logprobs"], return_text
        )
        r["meta_info"]["prefill_top_logprobs"] = await detokenize_top_logprobs_tokens(
            r["meta_info"]["prefill_top_logprobs"], return_text
        )
        r["meta_info"]["decode_top_logprobs"] = await detokenize_top_logprobs_tokens(
            r["meta_info"]["decode_top_logprobs"], return_text
        )

    if isinstance(obj.text, str):
        if obj.return_logprob:
            await convert_style(ret, obj.return_text_in_logprobs)
    else:
        for i, r in enumerate(ret):
            if obj.return_logprob[i]:
                await convert_style(r, obj.return_text_in_logprobs)


@app.post("/v1/completions")
async def v1_completions(raw_request: Request):
    request_json = await raw_request.json()
    request = CompletionRequest(**request_json)

    # TODO: Validate the request and return HTTPStatus.BAD_REQUEST if invalid.
    assert request.n == 1

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
    adapted_request.post_init()

    if adapted_request.stream:

        async def gnerate_stream_resp():
            stream_buffer = ""
            n_prev_token = 0
            async for content in stream_generator(adapted_request):
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

                    logprobs = await make_openai_style_logprobs(
                        prefill_token_logprobs=prefill_token_logprobs,
                        prefill_top_logprobs=prefill_top_logprobs,
                        decode_token_logprobs=content["meta_info"][
                            "decode_token_logprobs"
                        ][n_prev_token:],
                        decode_top_logprobs=content["meta_info"]["decode_top_logprobs"][
                            n_prev_token:
                        ],
                    )

                    n_prev_token = len(content["meta_info"]["decode_token_logprobs"])
                else:
                    logprobs = None

                delta = text[len(stream_buffer) :]
                stream_buffer = content["text"]
                choice_data = CompletionResponseStreamChoice(
                    index=0,
                    text=delta,
                    logprobs=logprobs,
                    finish_reason=None,
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
                yield f"data: {jsonify_pydantic_model(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gnerate_stream_resp(), media_type="text/event-stream")

    # Non-streaming response.
    ret = await generate_request(adapted_request)
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

        logprobs = await make_openai_style_logprobs(
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
        finish_reason=None,  # TODO(comaniac): Add finish reason.
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


@app.post("/v1/chat/completions")
async def v1_chat_completions(raw_request: Request):
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    # TODO: Validate the request and return HTTPStatus.BAD_REQUEST if invalid.
    assert request.n == 1

    # Prep the data needed for the underlying GenerateReqInput:
    #  - prompt: The full prompt string.
    #  - stop: Custom stop tokens.
    #  - image_data: None or a list of image strings (URLs or base64 strings).
    #    None skips any image processing in GenerateReqInput.
    if not isinstance(request.messages, str):
        # Apply chat template and its stop strings.
        if chat_template_name is None:
            # This flow doesn't support the full OpenAI spec.  Verify messages
            # has the right type before proceeding:
            for m in request.messages:
                if not isinstance(m.content, str):
                    raise HTTPException(
                        status_code=503,
                        detail="Structured content requests not supported with "
                        "HuggingFace Chat Templates. "
                        "Make sure the server specifies a sglang chat template.",
                    )
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
    adapted_request.post_init()

    if adapted_request.stream:

        async def gnerate_stream_resp():
            is_first = True

            stream_buffer = ""
            async for content in stream_generator(adapted_request):
                if is_first:
                    # First chunk with role
                    is_first = False
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {jsonify_pydantic_model(chunk)}\n\n"

                text = content["text"]
                delta = text[len(stream_buffer) :]
                stream_buffer = text
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(content=delta), finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    choices=[choice_data],
                    model=request.model,
                )
                yield f"data: {jsonify_pydantic_model(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gnerate_stream_resp(), media_type="text/event-stream")

    # Non-streaming response.
    ret = await generate_request(adapted_request)
    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=ret["text"]),
        finish_reason=None,  # TODO(comaniac): Add finish reason.
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


async def make_openai_style_logprobs(
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

            # Not Supported yet
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


def launch_server(server_args: ServerArgs, pipe_finish_writer):
    global tokenizer_manager

    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.show_time_cost:
        enable_show_time_cost()
    if server_args.disable_disk_cache:
        disable_cache()
    if server_args.enable_flashinfer:
        assert_pkg_version("flashinfer", "0.0.4")
    if server_args.chat_template:
        # TODO: replace this with huggingface transformers template
        load_chat_template_for_openai_api(server_args.chat_template)

    # Allocate ports
    server_args.port, server_args.additional_ports = allocate_init_ports(
        server_args.port, server_args.additional_ports, server_args.tp_size
    )
    port_args = PortArgs(
        tokenizer_port=server_args.additional_ports[0],
        router_port=server_args.additional_ports[1],
        detokenizer_port=server_args.additional_ports[2],
        nccl_port=server_args.additional_ports[3],
        model_rpc_ports=server_args.additional_ports[4:],
    )

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    proc_router = mp.Process(
        target=start_router_process,
        args=(
            server_args,
            port_args,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # Wait for the model to finish loading
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print(f"Initialization failed. router_init_state: {router_init_state}", flush=True)
        print(f"Initialization failed. detoken_init_state: {detoken_init_state}", flush=True)
        sys.exit(1)
    assert proc_router.is_alive() and proc_detoken.is_alive()

    if server_args.api_key and server_args.api_key != "":
        app.add_middleware(APIKeyValidatorMiddleware, api_key=server_args.api_key)

    def _wait_and_warmup():
        headers = {}
        url = server_args.url()
        if server_args.api_key:
            headers[API_KEY_HEADER_NAME] = server_args.api_key

        # Wait until the server is launched
        for _ in range(120):
            time.sleep(0.5)
            try:
                requests.get(url + "/get_model_info", timeout=5, headers=headers)
                success = True  # Set flag to True if request succeeds
                break
            except requests.exceptions.RequestException as e:
                pass

        # Send a warmup request
        try:
            res = requests.post(
                url + "/generate",
                json={
                    "text": "Say this is a warmup request.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200
        except Exception as e:
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(get_exception_traceback())
            print(f"Initialization failed. warmup error: {e}")
            raise e

        if pipe_finish_writer is not None:
            pipe_finish_writer.send("init ok")

    t = threading.Thread(target=_wait_and_warmup)
    t.start()
    try:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


class Runtime:
    def __init__(
        self,
        log_evel="error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_evel, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port, self.server_args.additional_ports, self.server_args.tp_size)

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(target=launch_server, args=(self.server_args, pipe_writer))
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError("Initialization failed. Please see the error messages above.")

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            psutil.wait_procs(children, timeout=5)
            parent.kill()
            parent.wait(timeout=5)
            self.pid = None

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def add_request(
        self,
        prompt: str,
        sampling_params,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
        }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)

    def __del__(self):
        self.shutdown()