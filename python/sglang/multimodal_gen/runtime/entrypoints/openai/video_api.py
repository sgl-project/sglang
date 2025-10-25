import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sglang.multimodal_gen.api.configs.sample.base import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    post_process_sample,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


# TODO: move this to `types.py`
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


class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"


# In-memory job store (simple, non-persistent)
# TODO: Encapsulate instead of direct call
VIDEO_JOBS: Dict[str, Dict[str, Any]] = {}
VIDEO_LOCK = asyncio.Lock()


def _build_sampling_params_from_request(
    request_id: str, request: VideoGenerationsRequest
) -> SamplingParams:
    width, height = _parse_size(request.size or "720x1280")
    seconds = request.seconds if request.seconds is not None else 4
    fps = 24  # TODO: allow user control of fps
    server_args = get_global_server_args()
    # TODO: should we cache this sampling_params?
    sampling_params = SamplingParams.from_pretrained(server_args.model_path)
    user_params = SamplingParams(
        request_id=request_id,
        prompt=request.prompt,
        num_frames=fps * seconds,
        fps=fps,
        width=width,
        height=height,
        output_file_name=request.input_reference,
        save_output=True,
    )
    sampling_params = sampling_params.from_user_sampling_params(user_params)
    sampling_params.log(server_args)
    sampling_params.set_output_file_ext()
    return sampling_params


# extract metadata which http_server needs to know
def _video_job_from_sampling(
    request_id: str, req: VideoGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": sampling.output_file_path(),
    }


async def _dispatch_job_async(job_id: str, batch: Req) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import scheduler_client

    try:
        result = await scheduler_client.forward([batch])
        post_process_sample(
            result.output[0],
            batch.data_type,
            batch.fps,
            batch.save_output,
            os.path.join(batch.output_path, batch.output_file_name),
        )
        async with VIDEO_LOCK:
            job = VIDEO_JOBS.get(job_id)
            if job is not None:
                job["status"] = "completed"
                job["progress"] = 100
                job["completed_at"] = int(time.time())
    except Exception as e:
        logger.error(f"{e}")
        async with VIDEO_LOCK:
            job = VIDEO_JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = {"message": str(e)}


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(request: VideoGenerationsRequest):
    logger.debug(f"Server received from create_video endpoint: {request=}")

    request_id = generate_request_id()
    sampling_params = _build_sampling_params_from_request(request_id, request)
    job = _video_job_from_sampling(request_id, request, sampling_params)
    async with VIDEO_LOCK:
        VIDEO_JOBS[request_id] = job

    # Build Req for scheduler
    batch = prepare_request(
        prompt=request.prompt,
        server_args=get_global_server_args(),
        sampling_params=sampling_params,
    )
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(_dispatch_job_async(request_id, batch))
    return VideoResponse(**job)


class VideoRemixRequest(BaseModel):
    prompt: str


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    # Normalize order
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    async with VIDEO_LOCK:
        jobs = list(VIDEO_JOBS.values())

    reverse = order != "asc"
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=reverse)

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    items = [VideoResponse(**j) for j in jobs]
    return VideoListResponse(data=items)


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    async with VIDEO_LOCK:
        job = VIDEO_JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(**job)


# TODO: support aborting a job.
@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    async with VIDEO_LOCK:
        job = VIDEO_JOBS.pop(video_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    # Mark as deleted in response semantics
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    async with VIDEO_LOCK:
        job = VIDEO_JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
