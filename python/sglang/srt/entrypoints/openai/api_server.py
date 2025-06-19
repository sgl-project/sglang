# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
SGLang OpenAI-Compatible API Server.

This file implements OpenAI-compatible HTTP APIs for the inference engine via FastAPI.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional

import numpy as np
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    register_disaggregation_server,
)
from sglang.srt.entrypoints.engine import Engine, _launch_subprocesses
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_prometheus_middleware,
    delete_directory,
    get_bool_env_var,
    kill_process_tree,
    set_uvicorn_logging_configs,
)
from sglang.srt.warmup import execute_warmups
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
class AppState:
    engine: Optional[Engine] = None
    server_args: Optional[ServerArgs] = None
    tokenizer_manager: Optional[TokenizerManager] = None
    scheduler_info: Optional[Dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.server_args.enable_metrics = True  # By default, we enable metrics

    server_args = app.state.server_args

    # Initialize engine
    logger.info(f"SGLang OpenAI server (PID: {os.getpid()}) is initializing...")

    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
    app.state.tokenizer_manager = tokenizer_manager
    app.state.scheduler_info = scheduler_info

    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Initialize engine state attribute to None for now
    app.state.engine = None

    if server_args.warmups is not None:
        await execute_warmups(
            server_args.warmups.split(","), app.state.tokenizer_manager
        )
        logger.info("Warmup ended")

    warmup_thread = getattr(app, "warmup_thread", None)
    if warmup_thread is not None:
        warmup_thread.start()

    yield

    # Lifespan shutdown
    if hasattr(app.state, "engine") and app.state.engine is not None:
        logger.info("SGLang engine is shutting down.")
        # Add engine cleanup logic here when implemented


# Fast API app with CORS enabled
app = FastAPI(
    lifespan=lifespan,
    # TODO: check where /openai.json is created or why we use this
    openapi_url=None if get_bool_env_var("DISABLE_OPENAPI_DOC") else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/health", methods=["GET"])
async def health() -> Response:
    """Health check. Used for readiness and liveness probes."""
    # In the future, this could check engine health more deeply
    # For now, if the server is up, it's healthy.
    return Response(status_code=200)


@app.api_route("/v1/models", methods=["GET"])
async def show_models():
    """Show available models. Currently, it returns the served model name.

    This endpoint is compatible with the OpenAI API standard.
    """
    served_model_names = [app.state.tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=app.state.tokenizer_manager.model_config.context_len,
            )
        )
    return ModelList(data=model_cards)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": app.state.tokenizer_manager.model_path,
        "tokenizer_path": app.state.tokenizer_manager.server_args.tokenizer_path,
        "is_generation": app.state.tokenizer_manager.is_generation,
    }
    return result


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    pass


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    pass


@app.post("/v1/embeddings")
async def openai_v1_embeddings(raw_request: Request):
    pass


@app.post("/v1/score")
async def v1_score_request(raw_request: Request):
    """Endpoint for the decoder-only scoring API. See Engine.score() for detailed documentation."""
    pass


# Additional API endpoints will be implemented in separate serving_*.py modules
# and mounted as APIRouters in future PRs


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection],
    image_token_text: str,
    launch_callback: Optional[Callable[[], None]] = None,
):
    return
    # TODO: Please wait until the /generate implementation is complete,
    # or confirm if modifications are needed before removing this.

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
    # TODO: Replace with OpenAI API
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(server_args.dp_size)]
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["input_ids"] = json_data["input_ids"][0]
    else:
        json_data["text"] = ["The capital city of France is"] * server_args.dp_size
        # TODO Workaround the bug that embedding errors for list of size 1
        if server_args.dp_size == 1:
            json_data["text"] = json_data["text"][0]

    # Debug dumping
    if server_args.debug_tensor_dump_input_file:
        json_data.pop("text", None)
        json_data["input_ids"] = np.load(
            server_args.debug_tensor_dump_input_file
        ).tolist()
        json_data["sampling_params"]["max_new_tokens"] = 0

    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
        else:
            logger.info(f"Start of prefill warmup ...")
            json_data = {
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
                # This is a hack to ensure fake transfer is enabled during prefill warmup
                # ensure each dp rank has a unique bootstrap_room during prefill warmup
                "bootstrap_room": [
                    i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                    for i in range(server_args.dp_size)
                ],
                "input_ids": [[0, 1, 2, 3]] * server_args.dp_size,
            }
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=1800,  # because of deep gemm precache is very long if not precache.
            )
            logger.info(
                f"End of prefill warmup with status {res.status_code}, resp: {res.json()}"
            )

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

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if server_args.pdlb_url is not None:
        register_disaggregation_server(
            server_args.disaggregation_mode,
            server_args.port,
            server_args.disaggregation_bootstrap_port,
            server_args.pdlb_url,
        )

    if launch_callback is not None:
        launch_callback()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang OpenAI-Compatible API Server")
    # Add arguments from ServerArgs. This allows reuse of existing CLI definitions.
    ServerArgs.add_cli_args(parser)
    # Potentially add server-specific arguments here in the future if needed

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # Store server_args in app.state for access in lifespan and endpoints
    app.state.server_args = server_args

    # Configure logging
    logging.basicConfig(
        level=server_args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            None,
            None,  # Never used
            None,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Start the server
        set_uvicorn_logging_configs()
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level.lower(),
            timeout_keep_alive=60,  # Increased keep-alive for potentially long requests
            loop="uvloop",  # Use uvloop for better performance if available
        )
    finally:
        warmup_thread.join()
