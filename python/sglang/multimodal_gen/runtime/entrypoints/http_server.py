# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.server_args import ServerArgs, prepare_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger


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

    app.include_router(image_api.router)
    app.include_router(video_api.router)

    app.state.server_args = server_args
    return app


if __name__ == "__main__":
    import uvicorn

    server_args = prepare_server_args([])
    configure_logger(server_args)
    app = create_app(server_args)
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_config=None,
        reload=False,  # Set to True during development for auto-reloading
    )
