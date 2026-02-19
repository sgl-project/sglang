# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import ORJSONResponse, Response
from pydantic import BaseModel

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VertexGenerateReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    StartProfileReq,
    StopProfileReq,
    build_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training import weights_api
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

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
    logger.info("FastAPI app is shutting down...")
    broker_task.cancel()
    async_scheduler_client.close()


# Health router
health_router = APIRouter()


@health_router.get("/health")
async def health():
    return {"status": "ok"}


@health_router.get("/models", deprecated=True)
async def get_models(request: Request):
    """
    Get information about the model served by this server.

    .. deprecated::
        Use /v1/models instead for OpenAI-compatible model discovery.
        This endpoint will be removed in a future version.
    """
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


class ProfileReqInput(BaseModel):
    """
    Request input for profile endpoints.
    Similar to LLM's ProfileReqInput for consistency.
    """

    # Directory to save traces. Uses SGLANG_TORCH_PROFILER_DIR env var if not provided.
    output_dir: Optional[str] = None
    # Activities to profile: ["CPU", "GPU"]
    activities: Optional[list[str]] = None
    # Whether to record stack traces
    with_stack: Optional[bool] = None
    # Whether to record tensor shapes
    record_shapes: Optional[bool] = None


class ProfileReqOutput(BaseModel):
    """Response output for profile endpoints."""

    success: bool
    message: str


# Global profiler state for online serving
_global_profiler_state = {
    "profiler": None,
    "is_profiling": False,
    "profile_id": None,
}


@health_router.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile(request: Request, obj: Optional[ProfileReqInput] = None):
    """Start profiling."""
    import time as time_module

    if _global_profiler_state["is_profiling"]:
        return Response(
            content="Profiling is already in progress. Call /stop_profile first.\n",
            status_code=400,
        )

    try:
        # Get parameters
        if obj is None:
            obj = ProfileReqInput()

        output_dir = obj.output_dir or envs.SGLANG_TORCH_PROFILER_DIR.get()

        profile_id = str(int(time_module.time()))

        with_stack = obj.with_stack or envs.SGLANG_PROFILE_WITH_STACK.get()
        record_shapes = obj.record_shapes or envs.SGLANG_PROFILE_RECORD_SHAPES.get()

        try:
            rank = get_world_rank()
        except Exception:
            logger.warning("Failed to get world rank, defaulting to 0")
            rank = 0

        # Lazy import to reduce import time (see issue #10492)
        from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

        http_profiler = SGLDiffusionProfiler(
            request_id=profile_id,
            rank=rank,
            full_profile=True,
            activities=obj.activities,
            num_steps=-1,
            num_inference_steps=100,
            log_dir=output_dir,
            with_stack=with_stack,
            record_shapes=record_shapes,
            is_host=True,
        )
        _global_profiler_state["profiler"] = http_profiler
        _global_profiler_state["profile_id"] = profile_id

        start_req = StartProfileReq(
            output_dir=output_dir,
            profile_id=profile_id,
            activities=obj.activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
        try:
            response = await async_scheduler_client.forward(start_req)
            if response.error:
                logger.warning(f"GPU Worker profiler warning: {response.error}")
        except Exception as e:
            logger.warning(f"Could not start GPU Worker profiler: {e}")

        _global_profiler_state["is_profiling"] = True

        logger.info(
            f"Profilers started with profile_id={profile_id}. " f"Output: {output_dir}"
        )

        return Response(
            content=f"Profiling started (profile_id={profile_id}). Traces will be saved to: {output_dir}\n",
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Failed to start profiler: {e}")
        return Response(
            content=f"Failed to start profiler: {str(e)}\n",
            status_code=500,
        )


@health_router.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    """Stop profiling and save traces."""
    if not _global_profiler_state["is_profiling"]:
        return Response(
            content="Profiling is not in progress. Call /start_profile first.\n",
            status_code=400,
        )

    errors = []
    profile_id = _global_profiler_state.get("profile_id", "unknown")

    try:
        # 1. Stop profiler in HTTP Server process
        profiler = _global_profiler_state["profiler"]
        if profiler is not None:
            profiler.stop(export_trace=True, dump_rank=None)  # Save for all ranks

        stop_req = StopProfileReq(export_trace=True)
        try:
            response = await async_scheduler_client.forward(stop_req)
            if response.error:
                errors.append(f"GPU Worker: {response.error}")
        except Exception as e:
            errors.append(f"GPU Worker: {str(e)}")

        _global_profiler_state["profiler"] = None
        _global_profiler_state["is_profiling"] = False
        _global_profiler_state["profile_id"] = None

        logger.info(f"Profilers stopped (profile_id={profile_id}). Traces saved.")

        if errors:
            return Response(
                content=f"Profiling stopped (profile_id={profile_id}) with warnings: {'; '.join(errors)}\n",
                status_code=200,
            )
        return Response(
            content=f"Profiling stopped (profile_id={profile_id}). Traces saved to SGLANG_TORCH_PROFILER_DIR.\n",
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Failed to stop profiler: {e}")
        _global_profiler_state["profiler"] = None
        _global_profiler_state["is_profiling"] = False
        _global_profiler_state["profile_id"] = None
        return Response(
            content=f"Failed to stop profiler: {str(e)}\n",
            status_code=500,
        )


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


async def forward_to_scheduler(
    req_obj: "Req",
    sp: SamplingParams,
):
    """Forwards request to scheduler and processes the result."""
    try:
        response = await async_scheduler_client.forward(req_obj)
        if response.output is None and response.output_file_paths is None:
            raise RuntimeError("Model generation returned no output.")

        if response.output_file_paths:
            output_file_path = response.output_file_paths[0]
        else:
            output_file_path = sp.output_file_path()
            save_outputs(
                [response.output[0]],
                sp.data_type,
                sp.fps,
                True,
                lambda _idx: output_file_path,
                audio=response.audio,
                audio_sample_rate=response.audio_sample_rate,
            )

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response if isinstance(response, dict) else vars(response)

        if output_file_path:
            logger.info("Processing output file: %s", output_file_path)
            b64_video = encode_video_to_base64(output_file_path)

            if b64_video:
                data["output"] = b64_video
                data.pop("video_data", None)
                data.pop("video_tensor", None)

        return make_serializable(data)

    except Exception as e:
        logger.error("Error during generation: %s", e, exc_info=True)
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

        sp = build_sampling_params(
            rid,
            prompt=inst.get("prompt") or inst.get("text"),
            image_path=inst.get("image") or inst.get("image_url"),
            seed=params.get("seed", DEFAULT_SEED),
            num_frames=params.get("num_frames"),
            fps=params.get("fps"),
            width=params.get("width"),
            height=params.get("height"),
            guidance_scale=params.get("guidance_scale"),
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
    app.include_router(weights_api.router)

    app.state.server_args = server_args
    return app
