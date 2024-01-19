"""SRT: SGLang Runtime"""
import asyncio
import json
import multiprocessing as mp
import sys
import threading
import time
from typing import List, Optional

# Fix a Python bug
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo
)
from sglang.srt.managers.router.manager import start_router_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import alloc_usable_network_port

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
    }
    return result

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
                delta = text[len(stream_buffer):]
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
                )
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        return StreamingResponse(gnerate_stream_resp(), media_type="text/event-stream")


    # Non-streaming response.
    ret = await generate_request(adapted_request)

    choice_data = CompletionResponseChoice(
        index=0,
        text=ret["text"],
        logprobs=None,
        finish_reason=None, # TODO(comaniac): Add finish reason.
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


def launch_server(server_args, pipe_finish_writer):
    global tokenizer_manager

    # Allocate ports
    can_use_ports = alloc_usable_network_port(
        num=4 + server_args.tp_size, used_list=(server_args.port,)
    )
    port_args = PortArgs(
        tokenizer_port=can_use_ports[0],
        router_port=can_use_ports[1],
        detokenizer_port=can_use_ports[2],
        nccl_port=can_use_ports[3],
        model_rpc_ports=can_use_ports[4:],
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
        print("router init state:", router_init_state)
        print("detoken init state:", detoken_init_state)
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    def launch_server():
        # Launch api server
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )

    t = threading.Thread(target=launch_server)
    t.start()

    if pipe_finish_writer:
        url = server_args.url()

        success = False
        for i in range(60):
            try:
                res = requests.get(url + "/get_model_info", timeout=5)
                success = True
                break
            except requests.exceptions.RequestException as e:
                time.sleep(1)

        if success:
            pipe_finish_writer.send("init ok")
        else:
            pipe_finish_writer.send(str(e))


class Runtime:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        load_format: str = "auto",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        mem_fraction_static: float = 0.9,
        tp_size: int = 1,
        model_mode: List[str] = (),
        schedule_heuristic: str = "lpm",
        random_seed: int = 42,
        log_level: str = "warning",
    ):
        host = "127.0.0.1"
        port = alloc_usable_network_port(1)[0]
        server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            host=host,
            port=port,
            load_format=load_format,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            mem_fraction_static=mem_fraction_static,
            tp_size=tp_size,
            model_mode=model_mode,
            schedule_heuristic=schedule_heuristic,
            random_seed=random_seed,
            log_level=log_level,
        )
        self.url = server_args.url()

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(target=launch_server, args=(server_args, pipe_writer))
        proc.start()
        self.pid = proc.pid

        init_state = pipe_reader.recv()
        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError("Launch failed")

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            parent = psutil.Process(self.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            psutil.wait_procs(children, timeout=5)
            parent.kill()
            parent.wait(timeout=5)
            self.pid = None

    def __del__(self):
        self.shutdown()
