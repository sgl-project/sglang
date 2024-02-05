"""SRT: SGLang Runtime"""
import asyncio
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
from sglang.srt.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.openai_protocol import (
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
    UsageInfo,
)
from sglang.srt.managers.router.manager import start_router_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import alloc_usable_network_port, handle_port_init

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


@app.get("/flush_cache")
async def flush_cache():
    await tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def stream_generator(obj):
    async for out in tokenizer_manager.generate_request(obj):
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
    return ret


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
        },
        stream=request.stream,
    )
    adapted_request.post_init()

    if adapted_request.stream:

        async def gnerate_stream_resp():
            stream_buffer = ""
            async for content in stream_generator(adapted_request):
                text = content["text"]
                prompt_tokens = content["meta_info"]["prompt_tokens"]
                completion_tokens = content["meta_info"]["completion_tokens"]

                delta = text[len(stream_buffer) :]
                stream_buffer = text
                choice_data = CompletionResponseStreamChoice(
                    index=0,
                    text=delta,
                    logprobs=None,
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
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gnerate_stream_resp(), media_type="text/event-stream")

    # Non-streaming response.
    ret = await generate_request(adapted_request)

    choice_data = CompletionResponseChoice(
        index=0,
        text=ret["text"],
        logprobs=None,
        finish_reason=None,  # TODO(comaniac): Add finish reason.
    )

    prompt_tokens = ret["meta_info"]["prompt_tokens"]
    completion_tokens = ret["meta_info"]["completion_tokens"]
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
                        detail="Structured content requests not supported with HuggingFace Chat Templates.  Make sure the server specifies a sglang chat template.",
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
                    yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

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
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
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


def launch_server(server_args, pipe_finish_writer):
    global tokenizer_manager
    global chat_template_name

    # Handle ports
    server_args.port, server_args.additional_ports = handle_port_init(
        server_args.port, server_args.additional_ports, server_args.tp_size
    )

    port_args = PortArgs(
        tokenizer_port=server_args.additional_ports[0],
        router_port=server_args.additional_ports[1],
        detokenizer_port=server_args.additional_ports[2],
        nccl_port=server_args.additional_ports[3],
        model_rpc_ports=server_args.additional_ports[4:],
    )

    # Load chat template if needed
    if server_args.chat_template is not None:
        print(f"Use chat template: {server_args.chat_template}")
        if not chat_template_exists(server_args.chat_template):
            if not os.path.exists(server_args.chat_template):
                raise RuntimeError(
                    f"Chat template {server_args.chat_template} is not a built-in template name "
                    "or a valid chat template file path."
                )
            with open(server_args.chat_template, "r") as filep:
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
            chat_template_name = server_args.chat_template

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
        print("router init state:", router_init_state)
        print("detoken init state:", detoken_init_state)
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    def _launch_server():
        # Launch api server
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )

    t = threading.Thread(target=_launch_server)
    t.start()

    url = server_args.url()
    for _ in range(60):
        time.sleep(1)
        try:
            requests.get(url + "/get_model_info", timeout=5)
            break
        except requests.exceptions.RequestException as e:
            pass
    else:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(str(e))
        else:
            print(e, flush=True)
        return

    # Warmup
    try:
        print("Warmup...", flush=True)
        res = requests.post(
            url + "/generate",
            json={
                "text": "Say this is a warmup request.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            timeout=60,
        )
        print(f"Warmup done. model response: {res.json()['text']}")
    except requests.exceptions.RequestException as e:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(str(e))
        else:
            print(e, flush=True)
        return

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("init ok")


class Runtime:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        load_format: str = "auto",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        mem_fraction_static: float = ServerArgs.mem_fraction_static,
        max_prefill_num_token: int = ServerArgs.max_prefill_num_token,
        tp_size: int = 1,
        model_mode: List[str] = (),
        schedule_heuristic: str = "lpm",
        random_seed: int = 42,
        log_level: str = "error",
        port: Optional[int] = None,
        additional_ports: Optional[Union[List[int], int]] = None,
    ):
        host = "127.0.0.1"
        port, additional_ports = handle_port_init(port, additional_ports, tp_size)
        self.server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            host=host,
            port=port,
            additional_ports=additional_ports,
            load_format=load_format,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            mem_fraction_static=mem_fraction_static,
            max_prefill_num_token=max_prefill_num_token,
            tp_size=tp_size,
            model_mode=model_mode,
            schedule_heuristic=schedule_heuristic,
            random_seed=random_seed,
            log_level=log_level,
        )

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
            raise RuntimeError("Launch failed. Please see the error messages above.")

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
    ) -> None:
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
