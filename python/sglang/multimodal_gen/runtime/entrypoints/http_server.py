# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import os
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.srt.managers.io_struct import VertexGenerateReqInput

DEFAULT_SEED = 1024
VERTEX_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sglang.multimodal_gen.runtime.scheduler_client import (
        async_scheduler_client,
        run_zeromq_broker,
    )

    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    async_scheduler_client.initialize(server_args)

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))

    yield

    # On shutdown
    print("FastAPI app is shutting down...")
    broker_task.cancel()
    async_scheduler_client.close()


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
    }

    if model_info:
        response["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        response["pipeline_class"] = model_info.pipeline_cls.__name__

    return response


@health_router.get("/health_generate")
async def health_generate():
    # TODO : health generate endpoint
    return {"status": "ok"}


def make_serializable(obj):
    """Recursively converts Tensors to None for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def encode_video_to_base64(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def forward_to_scheduler(req_obj, sp):
    """Forwards request to scheduler and processes the result."""
    try:
        response = await async_scheduler_client.forward(req_obj)
        if response.output is None:
            raise RuntimeError("Model generation returned no output.")

        output_file_path = sp.output_file_path()
        post_process_sample(
            sample=response.output[0],
            data_type=sp.data_type,
            fps=sp.fps or 24,
            save_output=True,
            save_file_path=output_file_path,
        )

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response if isinstance(response, dict) else vars(response)

        if output_file_path:
            print(f"Processing output file: {output_file_path}")
            b64_video = encode_video_to_base64(output_file_path)

            if b64_video:
                data["output"] = b64_video
                data.pop("video_data", None)
                data.pop("video_tensor", None)

        return make_serializable(data)

    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": str(e)}


vertex_router = APIRouter()


@vertex_router.post(VERTEX_ROUTE)
async def vertex_generate(vertex_req: VertexGenerateReqInput):
    if not vertex_req.instances:
        return ORJSONResponse({"predictions": []})

    server_args = get_global_server_args()
    params = vertex_req.parameters or {}

    futures = []

    for inst in vertex_req.instances:
        rid = f"vertex_{uuid.uuid4()}"

        prompt = inst.get("prompt") or inst.get("text")
        image_input = inst.get("image") or inst.get("image_url")
        seed_val = params.get("seed", DEFAULT_SEED)

        sp = SamplingParams.from_user_sampling_params_args(
            model_path=server_args.model_path,
            request_id=rid,
            prompt=prompt,
            image_path=image_input,
            num_frames=params.get("num_frames"),
            fps=params.get("fps"),
            width=params.get("width"),
            height=params.get("height"),
            guidance_scale=params.get("guidance_scale"),
            seed=seed_val,
            server_args=server_args,
            save_output=params.get("save_output"),
        )

        backend_req = prepare_request(server_args, sampling_params=sp)
        futures.append(forward_to_scheduler(backend_req, sp))

    results = await asyncio.gather(*futures)

    return ORJSONResponse({"predictions": results})


def create_app(server_args: ServerArgs):
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router)
    app.include_router(vertex_router)

    from sglang.multimodal_gen.runtime.entrypoints.openai import common_api

    app.include_router(common_api.router)
    app.include_router(image_api.router)
    app.include_router(video_api.router)

    app.state.server_args = server_args
    return app
