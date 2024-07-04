"""
The entry point of inference server.
SRT = SGLang Runtime.
"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from http import HTTPStatus
from typing import Dict, Optional

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.constrained import disable_cache
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.controller.manager_multi import (
    start_controller_process as start_controller_process_multi,
)
from sglang.srt.managers.controller.manager_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.controller.tp_worker import ModelTpService
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api_adapter import (
    load_chat_template_for_openai_api,
    v1_chat_completions,
    v1_completions,
)
from sglang.srt.server_args import ModelPortArgs, PortArgs, ServerArgs
from sglang.srt.utils import (
    API_KEY_HEADER_NAME,
    APIKeyValidatorMiddleware,
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
    send_addrs_to_rank_0,
    receive_addrs,
    start_rpyc_service_process,
)
from sglang.utils import get_exception_traceback


logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None


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
    tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def generate_request(obj: GenerateReqInput, request: Request):
    if obj.stream:

        async def stream_results():
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await tokenizer_manager.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(tokenizer_manager, raw_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, raw_request)


def launch_server(server_args: ServerArgs, pipe_finish_writer, model_overide_args=None):
    global tokenizer_manager

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.show_time_cost:
        enable_show_time_cost()
    if server_args.disable_disk_cache:
        disable_cache()
    if not server_args.disable_flashinfer:
        assert_pkg_version("flashinfer", "0.0.8", "Please uninstall the old version and "
                           "reinstall the latest version by following the instructions "
                           "at https://docs.flashinfer.ai/installation.html.")
    if server_args.chat_template:
        # TODO: replace this with huggingface transformers template
        load_chat_template_for_openai_api(server_args.chat_template)

    # Allocate ports
    assert server_args.tp_size % server_args.nnodes == 0
    tp_size_local = server_args.tp_size // server_args.nnodes
    server_args.port, server_args.additional_ports = allocate_init_ports(
        server_args.port,
        server_args.additional_ports,
        tp_size_local,
        server_args.dp_size,
    )

    ports = server_args.additional_ports
    model_port_args = []
    for i in range(server_args.dp_size):
        model_port_args.append(
            ModelPortArgs(
                nccl_port=ports[3 + i * (tp_size_local + 1)],
                model_tp_ips=[None] * tp_size_local,
                model_tp_ports=ports[3 + i * (tp_size_local + 1) + 1 : 3 + (i + 1) * (tp_size_local + 1)],
            )
        )
    port_args = PortArgs(
        tokenizer_port=ports[0],
        router_port=ports[1],
        detokenizer_port=ports[2],
        model_port_args=model_port_args,
    )

    # TODO multi-node dp is not supported
    assert not (server_args.dp_size > 1 and server_args.node_rank is not None)
    if server_args.nnodes > 1:
        if server_args.node_rank != 0:
            send_addrs_to_rank_0(model_port_args[0], server_args)
        else:
            receive_addrs(model_port_args[0], server_args)
        for i in range(tp_size_local):
            start_rpyc_service_process(ModelTpService, model_port_args[0].model_tp_ports[i])
        if server_args.node_rank != 0:
            logger.info(f"[node_rank={server_args.node_rank}]: Listen for connections...")
            while True:
                pass

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args, model_overide_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    if server_args.dp_size == 1:
        start_process = start_controller_process_single
    else:
        start_process = start_controller_process_multi
    proc_router = mp.Process(
        target=start_process,
        args=(server_args, port_args, pipe_router_writer, model_overide_args),
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
        print(
            f"Initialization failed. router_init_state: {router_init_state}", flush=True
        )
        print(
            f"Initialization failed. detoken_init_state: {detoken_init_state}",
            flush=True,
        )
        sys.exit(1)
    assert proc_router.is_alive() and proc_detoken.is_alive()

    if server_args.api_key and server_args.api_key != "":
        app.add_middleware(APIKeyValidatorMiddleware, api_key=server_args.api_key)

    # Send a warmup request
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
                break
            except requests.exceptions.RequestException:
                pass

        # Send a warmup request
        try:
            for _ in range(server_args.dp_size):
                res = requests.post(
                    url + "/generate",
                    json={
                        "text": "The capital city of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                        },
                    },
                    headers=headers,
                    timeout=600,
                )
                assert res.status_code == 200
        except Exception as e:
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(get_exception_traceback())
            print(f"Initialization failed. warmup error: {e}", flush=True)
            raise e

        logger.info("The server is fired up and ready to roll!")
        if pipe_finish_writer is not None:
            pipe_finish_writer.send("init ok")

    t = threading.Thread(target=_wait_and_warmup)
    t.start()

    # Listen for requests
    try:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


class Runtime:
    """
    A wrapper for the server.
    This is used for launching the server in a python program without
    using the commond line interface.
    """

    def __init__(
        self,
        log_level: str = "error",
        model_overide_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.tp_size,
            self.server_args.dp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer, model_overide_args),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

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
        sampling_params: Dict,
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
