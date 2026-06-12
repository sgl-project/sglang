"""Native SGLang-D API for diffusion generation with extended metadata.

Exposes trajectory data (latents, timesteps, decoded frames) and log_probs
that the OpenAI-compatible endpoints intentionally omit. Intended for RL
training pipelines and workloads that need intermediate diffusion outputs.
"""

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    process_generation_batch,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1/diffusion", tags=["diffusion-native"])
logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DiffusionGenerateRequest(BaseModel):
    """Native SGLang-D generation request with trajectory and log_prob support."""

    prompt: str
    negative_prompt: Optional[str] = None
    image_path: Optional[str] = None

    # Dimensions
    size: Optional[str] = None  # "WxH" convenience string
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None

    # Generation parameters
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    num_outputs_per_prompt: Optional[int] = 1
    enable_teacache: Optional[bool] = False

    # Extended metadata flags (default False — no latency impact when unused)
    get_latents: bool = False
    get_log_probs: bool = False

    # Output control
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    enable_frame_interpolation: Optional[bool] = False
    frame_interpolation_exp: Optional[int] = 1
    frame_interpolation_scale: Optional[float] = 1.0
    frame_interpolation_model_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None


class TrajectoryData(BaseModel):
    """Serialized trajectory arrays as base64-encoded ``.npy`` blobs.

    Client deserialization::

        import base64, io, numpy as np
        arr = np.load(io.BytesIO(base64.b64decode(blob)))
    """

    latents: Optional[str] = None
    latents_shape: Optional[List[int]] = None
    latents_dtype: Optional[str] = None
    timesteps: Optional[List[str]] = None
    log_probs: Optional[str] = None
    log_probs_shape: Optional[List[int]] = None


class DiffusionGenerateResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    output_b64: Optional[str] = None
    output_format: Optional[str] = None
    file_path: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None
    trajectory: Optional[TrajectoryData] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_to_b64_npy(t: torch.Tensor) -> str:
    """Serialize a torch.Tensor to a base64-encoded ``.npy`` blob."""
    arr = t.detach().cpu().float().numpy()
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_trajectory(result) -> Optional[TrajectoryData]:
    """Extract trajectory fields from an OutputBatch into serialized form."""
    has_latents = result.trajectory_latents is not None
    has_log_probs = getattr(result, "trajectory_log_probs", None) is not None
    if not has_latents and not has_log_probs:
        return None

    latents_b64 = None
    latents_shape = None
    latents_dtype = None
    if has_latents:
        latents_b64 = _tensor_to_b64_npy(result.trajectory_latents)
        latents_shape = list(result.trajectory_latents.shape)
        latents_dtype = str(result.trajectory_latents.dtype)

    timesteps_b64 = None
    if result.trajectory_timesteps is not None:
        timesteps_b64 = [_tensor_to_b64_npy(t) for t in result.trajectory_timesteps]

    log_probs_b64 = None
    log_probs_shape = None
    if has_log_probs:
        log_probs_b64 = _tensor_to_b64_npy(result.trajectory_log_probs)
        log_probs_shape = list(result.trajectory_log_probs.shape)

    return TrajectoryData(
        latents=latents_b64,
        latents_shape=latents_shape,
        latents_dtype=latents_dtype,
        timesteps=timesteps_b64,
        log_probs=log_probs_b64,
        log_probs_shape=log_probs_shape,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=DiffusionGenerateResponse)
async def generate(request: DiffusionGenerateRequest):
    """Generate image/video with optional trajectory metadata."""
    request_id = generate_request_id()
    server_args = get_global_server_args()

    with temp_dir_if_disabled(server_args.output_path) as output_dir:
        # Build SamplingParams with output_path set upfront so _adjust()
        # correctly sets save_output=True and generates a valid file path.
        sampling = build_sampling_params(
            request_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image_path=request.image_path,
            size=request.size,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            fps=request.fps,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            guidance_scale_2=request.guidance_scale_2,
            true_cfg_scale=request.true_cfg_scale,
            num_outputs_per_prompt=request.num_outputs_per_prompt,
            enable_teacache=request.enable_teacache,
            return_trajectory_latents=request.get_latents,
            return_trajectory_decoded=request.get_latents,
            output_path=output_dir,
            output_file_name=request_id,
            output_quality=request.output_quality,
            output_compression=request.output_compression,
            enable_frame_interpolation=request.enable_frame_interpolation,
            frame_interpolation_exp=request.frame_interpolation_exp,
            frame_interpolation_scale=request.frame_interpolation_scale,
            frame_interpolation_model_path=request.frame_interpolation_model_path,
        )

        batch = prepare_request(server_args=server_args, sampling_params=sampling)
        if request.diffusers_kwargs:
            batch.extra["diffusers_kwargs"] = request.diffusers_kwargs

        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )

        # Read output file as base64
        output_b64 = None
        file_path = save_file_path_list[0] if save_file_path_list else None
        output_format = None
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                output_b64 = base64.b64encode(f.read()).decode("ascii")
            ext = os.path.splitext(file_path)[1].lstrip(".")
            output_format = ext if ext else None

        is_persistent = server_args.output_path is not None

        # Serialize trajectory data from OutputBatch
        trajectory = _build_trajectory(result)

    return DiffusionGenerateResponse(
        id=request_id,
        output_b64=output_b64,
        output_format=output_format,
        file_path=file_path if is_persistent else None,
        peak_memory_mb=(
            result.peak_memory_mb
            if result.peak_memory_mb and result.peak_memory_mb > 0
            else None
        ),
        inference_time_s=(
            result.metrics.total_duration_s
            if result.metrics and result.metrics.total_duration_s > 0
            else None
        ),
        trajectory=trajectory,
    )
