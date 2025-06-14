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
import os
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.metrics.func_timer import enable_func_timer
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_prometheus_middleware,
    get_bool_env_var,
    kill_process_tree,
    set_prometheus_multiproc_dir,
)

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


# Store global states
class AppState:
    engine = None
    server_args = None
    tokenizer_manager = None
    scheduler_info = None


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

    # TODO: Add warmup logic here

    yield

    # Lifespan shutdown
    if hasattr(app.state, "engine") and app.state.engine is not None:
        logger.info("SGLang engine is shutting down.")
        # Add engine cleanup logic here when implemented

    # TODO: Implement a proper and graceful shutdown logic
    kill_process_tree(os.getpid())


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


@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    pass


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    pass


@app.post("/v1/embeddings")
async def openai_v1_embeddings(raw_request: Request):
    pass


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    pass


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    pass


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    pass


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve
    pass


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve-contents
    pass


@app.post("/v1/score")
async def v1_score_request(raw_request: Request):
    """Endpoint for the decoder-only scoring API. See Engine.score() for detailed documentation."""
    pass


# Additional API endpoints will be implemented in separate serving_*.py modules
# and mounted as APIRouters in future PRs

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

    # Start the server
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level.lower(),
        timeout_keep_alive=60,  # Increased keep-alive for potentially long requests
        loop="uvloop",  # Use uvloop for better performance if available
    )
