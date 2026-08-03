import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    file_path: str | None = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageResponseData]
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None


class ImageGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    model: str | None = None
    n: int | None = 1
    quality: str | None = "auto"
    response_format: str | None = "url"  # url | b64_json
    size: str | None = "1024x1024"  # e.g., 1024x1024
    style: str | None = "vivid"
    background: str | None = "auto"  # transparent | opaque | auto
    output_format: str | None = None  # png | jpeg | webp
    user: str | None = None
    # SGLang extensions
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    true_cfg_scale: float | None = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    seed: int | list[int] | None = None
    generator_device: str | None = "cuda"
    negative_prompt: str | None = None
    output_quality: str | None = "default"
    output_compression: int | None = None
    enable_teacache: bool | None = False
    max_sequence_length: int | None = None
    flow_shift: float | None = None
    # Upscaling
    enable_upscaling: bool | None = False
    upscaling_model_path: str | None = None
    upscaling_scale: int | None = 4
    diffusers_kwargs: dict[str, Any] | None = None  # kwargs for diffusers backend
    # Client-supplied id for idempotent dispatch, status, and cancel (job control)
    request_id: str | None = None
    # Performance profiling
    perf_dump_path: str | None = None
    # Progressive resolution generation
    progressive_mode: str | None = None
    progressive_levels: int | None = None
    progressive_delta: float | None = None


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    num_outputs: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None
    action: Optional[Dict[str, Any]] = None


class VideoGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str
    input_reference: Optional[str] = None
    reference_url: Optional[str] = None
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    model: Optional[str] = None
    n: Optional[int] = 1
    num_outputs_per_prompt: Optional[int] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[Union[int, List[int]]] = None
    generator_device: Optional[str] = "cuda"
    # SGLang extensions
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    negative_prompt: Optional[str] = None
    max_sequence_length: Optional[int] = None
    flow_shift: Optional[float] = None
    enable_teacache: Optional[bool] = False
    # Frame interpolation
    enable_frame_interpolation: Optional[bool] = False
    frame_interpolation_exp: Optional[int] = 1  # 1=2×, 2=4×
    frame_interpolation_scale: Optional[float] = 1.0
    frame_interpolation_model_path: Optional[str] = None
    # Upscaling
    enable_upscaling: Optional[bool] = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: Optional[int] = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    output_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend
    # Performance profiling
    perf_dump_path: Optional[str] = None
    profile: Optional[bool] = False
    num_profiled_timesteps: Optional[int] = None
    profile_all_stages: Optional[bool] = False


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str


class RealtimeVideoGenerationsRequest(VideoGenerationsRequest):
    type: Literal["init"]
    # WebSocket does not support multipart/form-data image uploads
    first_frame: bytes | str | None = None
    condition_inputs: dict[str, Any] | None = None
    max_chunks: int | None = Field(default=None, ge=1)
    seed: int | None = 42
    guidance_scale: float | None = 1.0
    size: str | None = "832x480"
    profile: bool | None = False
    num_profiled_timesteps: int | None = None
    profile_all_stages: bool | None = False
    realtime_output_format: Literal["raw", "webp", "jpeg"] | None = None
    realtime_preview_max_width: int | None = None
    realtime_output_pacing: bool | None = False
    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None


class RealtimeEvent(BaseModel):
    type: Literal["event"]
    kind: str
    payload: Any = None
    event_id: int | None = None


# Mesh API protocol models
class MeshResponse(BaseModel):
    id: str
    object: str = "mesh"
    model: str = ""
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    format: str = "glb"
    url: str | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error: dict[str, Any] | None = None
    file_path: str | None = None
    file_size_bytes: int | None = None
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None


class MeshGenerationsRequest(BaseModel):
    prompt: str = "generate 3d mesh"
    input_image: str | None = None
    model: str | None = None
    seed: int | list[int] | None = None
    generator_device: str | None = "cuda"
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    output_format: str | None = "glb"


class MeshListResponse(BaseModel):
    data: list[MeshResponse]
    object: str = "list"


@dataclass
class BaseReq(ABC):
    rid: str | list[str] | None = field(default=None, kw_only=True)
    http_worker_ipc: str | None = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: list[dict]
    parameters: dict | None = None
