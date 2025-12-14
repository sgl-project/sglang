# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request

from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sglang.multimodal_gen.runtime.scheduler_client import (
        run_zeromq_broker,
        scheduler_client,
    )

    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    scheduler_client.initialize(server_args)

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))

    yield

    # On shutdown
    print("FastAPI app is shutting down...")
    broker_task.cancel()
    scheduler_client.close()


# Health router
health_router = APIRouter()


@health_router.get("/health")
async def health():
    return {"status": "ok"}


@health_router.get("/models")
async def get_models(request: Request):
    """Get information about the model served by this server."""
    from sglang.multimodal_gen.registry import get_model_info

    server_args: ServerArgs = request.app.state.server_args
    model_info = get_model_info(server_args.model_path)

    response = {
        "model_path": server_args.model_path,
        "num_gpus": server_args.num_gpus,
    }

    if task_type_enum := getattr(server_args.pipeline_config, "task_type", None):
        response["task_type"] = task_type_enum.name

    if workload_type_enum := getattr(server_args, "workload_type", None):
        response["workload_type"] = workload_type_enum.value

    if model_info:
        if pipeline_name := getattr(model_info.pipeline_cls, "pipeline_name", None):
            response["pipeline_name"] = pipeline_name
        response["pipeline_class"] = model_info.pipeline_cls.__name__

    if dit_precision := getattr(server_args.pipeline_config, "dit_precision", None):
        response["dit_precision"] = dit_precision

    if vae_precision := getattr(server_args.pipeline_config, "vae_precision", None):
        response["vae_precision"] = vae_precision

    return response


@health_router.get("/health_generate")
async def health_generate():
    # TODO : health generate endpoint
    return {"status": "ok"}


def create_app(server_args: ServerArgs):
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router)

    from sglang.multimodal_gen.runtime.entrypoints.openai import common_api

    app.include_router(common_api.router)
    app.include_router(image_api.router)
    app.include_router(video_api.router)

    app.state.server_args = server_args
    return app
