# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import os
import signal
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import httpx
import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import (
    image_api,
    video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VertexGenerateReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training import (
    rollout_api,
    weights_api,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.server_warmup import (
    run_async_client_warmup,
    should_run_synthetic_server_warmup,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils.json_response import orjson_response
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

VERTEX_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate")
SERVER_WARMUP_BYPASS_PATHS = (
    "/health",
    "/health_generate",
    "/model_info",
    "/server_info",
)


async def _wait_until_http_ready(server_args: ServerArgs) -> None:
    """for server warmup"""
    health_url = f"{server_args.url()}/health"
    # Probe the local server directly: a loopback readiness check must never be
    # routed through an HTTP proxy. trust_env=False also avoids crashing startup
    # on a malformed proxy env var, since httpx parses *_PROXY/NO_PROXY when the
    # client is constructed (raising httpx.InvalidURL before any request). See #28493.
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(120):
            try:
                response = await client.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"HTTP server did not become ready at {health_url}")


async def _run_server_warmup_after_http_ready(
    server_args: ServerArgs, warmup_done: asyncio.Event
) -> None:
    try:
        if not should_run_synthetic_server_warmup(server_args):
            warmup_done.set()
            return

        await _wait_until_http_ready(server_args)

        await run_async_client_warmup(
            server_args,
            async_scheduler_client.forward,
            fail_open=server_args.warmup_resolutions is None,
        )
        logger.info("The server is fired up and ready to roll!")
        warmup_done.set()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error("Server warmup failed; aborting startup: %s", e, exc_info=True)
        os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sglang.multimodal_gen.runtime.scheduler_client import (
        async_scheduler_client,
        run_zeromq_broker,
    )

    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    async_scheduler_client.initialize(server_args)
    warmup_done = asyncio.Event()
    app.state.server_warmup_done = warmup_done

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))
    warmup_task = None
    if server_args.server_warmup:
        warmup_task = asyncio.create_task(
            _run_server_warmup_after_http_ready(server_args, warmup_done)
        )
    else:
        warmup_done.set()

    try:
        yield
    finally:
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

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
    model_info = get_model_info(server_args.model_path, model_id=server_args.model_id)

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


@health_router.get("/server_info")
async def server_info_endpoint(request: Request):
    """Get server information.

    Returns fields compatible with the LLM engine's /server_info so that
    the model gateway can discover diffusion workers.
    """
    server_args: ServerArgs = request.app.state.server_args

    return {
        "model_path": server_args.model_path,
        "served_model_name": server_args.model_id or server_args.model_path,
        "tp_size": server_args.tp_size,
        "dp_size": server_args.dp_size,
        "version": __version__,
    }


@health_router.get("/model_info")
async def model_info_endpoint(request: Request):
    """Get model information.

    Returns fields compatible with the LLM engine's /model_info so that
    the model gateway can detect capabilities for diffusion workers.
    """
    from sglang.multimodal_gen.registry import get_model_info

    server_args: ServerArgs = request.app.state.server_args
    task_type = server_args.pipeline_config.task_type

    try:
        registry_info = get_model_info(
            server_args.model_path,
            backend=server_args.backend,
            model_id=server_args.model_id,
        )
    except Exception:
        logger.warning("Failed to resolve model info from registry", exc_info=True)
        registry_info = None

    return {
        # Fields consumed by the model gateway for worker discovery
        "model_path": server_args.model_path,
        "is_generation": True,
        "model_type": "diffusion",
        "architectures": (
            [registry_info.pipeline_cls.__name__] if registry_info else None
        ),
        # Fields matching the LLM engine's /model_info shape
        "has_image_understanding": task_type.accepts_image_input(),
        "has_audio_understanding": False,
        # Diffusion-specific fields
        "task_type": task_type.name,
        "is_image_gen": task_type.is_image_gen(),
    }


@health_router.get("/health_generate")
async def health_generate():
    # TODO : health generate endpoint
    return {"status": "ok"}


@health_router.get("/stats")
async def stats_endpoint(request: Request):
    """Get runtime statistics including disagg pipeline metrics.

    Returns queue depth, request counts, latency, throughput, etc.
    Sends a GetDisaggStatsReq to the scheduler via ZMQ and returns the result.
    """
    from sglang.multimodal_gen.runtime.entrypoints.utils import GetDisaggStatsReq

    server_args: ServerArgs = request.app.state.server_args
    response: dict = {
        "status": "ok",
        "model_path": server_args.model_path,
    }

    # Query the scheduler for disagg metrics
    try:
        stats_response = await async_scheduler_client.forward(GetDisaggStatsReq())
        if hasattr(stats_response, "output") and stats_response.output is not None:
            response["disagg"] = stats_response.output
    except Exception as e:
        response["disagg"] = {"error": str(e)}

    return response


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
                enable_frame_interpolation=sp.enable_frame_interpolation,
                frame_interpolation_exp=sp.frame_interpolation_exp,
                frame_interpolation_scale=sp.frame_interpolation_scale,
                frame_interpolation_model_path=sp.frame_interpolation_model_path,
                enable_upscaling=sp.enable_upscaling,
                upscaling_model_path=sp.upscaling_model_path,
                upscaling_scale=sp.upscaling_scale,
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
        return orjson_response({"predictions": []})

    server_args = get_global_server_args()
    params = vertex_req.parameters or {}

    futures = []

    for inst in vertex_req.instances:
        rid = f"vertex_{uuid.uuid4()}"

        sp = build_sampling_params(
            rid,
            prompt=inst.get("prompt") or inst.get("text"),
            image_path=inst.get("image") or inst.get("image_url"),
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

    return orjson_response({"predictions": results})


def create_app(server_args: ServerArgs):
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def wait_for_server_warmup(request: Request, call_next):
        warmup_done = getattr(request.app.state, "server_warmup_done", None)
        if (
            warmup_done is not None
            and not warmup_done.is_set()
            and request.url.path not in SERVER_WARMUP_BYPASS_PATHS
        ):
            await warmup_done.wait()
        return await call_next(request)

    app.include_router(health_router)
    app.include_router(vertex_router)

    from sglang.multimodal_gen.runtime.entrypoints.openai import common_api, mesh_api

    app.include_router(common_api.router)
    app.include_router(image_api.router)
    app.include_router(video_api.router)
    app.include_router(realtime_video_api.router)
    app.include_router(mesh_api.router)
    app.include_router(weights_api.router)
    app.include_router(rollout_api.router)

    from sglang.multimodal_gen.runtime.entrypoints import diffusion_api

    app.include_router(diffusion_api.router)

    app.state.server_args = server_args
    return app
