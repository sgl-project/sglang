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
        "task_type": server_args.pipeline_config.task_type.name,
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
        "workload_type": server_args.workload_type.value,
    }

    if model_info:
        response["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        response["pipeline_class"] = model_info.pipeline_cls.__name__

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
