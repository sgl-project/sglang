import atexit
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from typing import Dict, List, Optional, Union

import aiohttp
import requests
import uvicorn
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.server import Engine
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    add_prometheus_middleware,
    delete_directory,
    is_port_available,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Runtime:
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the commond line interface.

    It is mainly used for the frontend language.
    You should use the Engine class above if you want to do normal offline processing.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        # Pre-allocate ports
        for port in range(self.server_args.port, 40000):
            if is_port_available(port):
                break
        self.server_args.port = port

        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "ready":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            kill_process_tree(self.pid)
            self.pid = None

    def cache_prefix(self, prefix: str):
        self.endpoint.cache_prefix(prefix)

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        if self.server_args.skip_tokenizer_init:
            json_data = {
                "input_ids": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        else:
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
                        if "text" in data:
                            cur = data["text"][pos:]
                            if cur:
                                yield cur
                            pos += len(cur)
                        else:
                            yield data

    add_request = async_generate

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "lora_path": lora_path,
        }
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)
        response = requests.post(
            self.url + "/generate",
            json=json_data,
        )
        return json.dumps(response.json())

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        json_data = {"text": prompt}
        response = requests.post(self.url + "/encode", json=json_data)
        return json.dumps(response.json())

    async def get_server_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/get_server_info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise RuntimeError(
                        f"Failed to get server info. {error_data['error']['message']}"
                    )

    def __del__(self):
        self.shutdown()


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """
    Launch SRT (SGLang Runtime) Server

    The SRT server consists of an HTTP server and the SRT engine.

    1. HTTP server: A FastAPI server that routes requests to the engine.
    2. SRT engine:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """
    from sglang.srt.server import fastapi_app
    from sglang.srt.server.fastapi_app import app

    engine = Engine(server_args=server_args)
    fastapi_app.setup_global_state(engine=engine)

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            tokenizer_manager.image_token_id,
        ),
    )
    t.start()

    try:
        # Update logging configs
        set_uvicorn_logging_configs()

        # Listen for HTTP requests
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


def _wait_and_warmup(server_args, pipe_finish_writer, image_token_text):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    model_info = res.json()

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [10, 11, 12]
    else:
        json_data["text"] = "The capital city of France is"

    try:
        for _ in range(server_args.dp_size):
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return

    # Debug print
    # logger.info(f"{res.json()=}")

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)
