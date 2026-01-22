import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    file_path: Optional[str] = None


class ImageResponse(BaseModel):
    id: str
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "auto"
    response_format: Optional[str] = "url"  # url | b64_json
    size: Optional[str] = "1024x1024"  # e.g., 1024x1024
    style: Optional[str] = "vivid"
    background: Optional[str] = "auto"  # transparent | opaque | auto
    output_format: Optional[str] = None  # png | jpeg | webp
    user: Optional[str] = None
    # SGLang extensions
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    negative_prompt: Optional[str] = None
    enable_teacache: Optional[bool] = False
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend


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
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: Optional[str] = None
    model: Optional[str] = None
    seconds: Optional[int] = 4
    size: Optional[str] = ""
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"
    # SGLang extensions
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    true_cfg_scale: Optional[float] = (
        None  # for CFG vs guidance distillation (e.g., QwenImage)
    )
    negative_prompt: Optional[str] = None
    enable_teacache: Optional[bool] = False
    output_path: Optional[str] = None
    diffusers_kwargs: Optional[Dict[str, Any]] = None  # kwargs for diffusers backend


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str


@dataclass
class BaseReq(ABC):
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None
