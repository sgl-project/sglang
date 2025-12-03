import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Image API protocol models
class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]


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


# Video API protocol models
class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = "720x1280"
    seconds: str = "4"
    quality: str = "standard"
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    input_reference: Optional[str] = None
    model: Optional[str] = None
    seconds: Optional[int] = 4
    size: Optional[str] = "720x1280"
    fps: Optional[int] = None
    num_frames: Optional[int] = None


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


class VideoRemixRequest(BaseModel):
    prompt: str
