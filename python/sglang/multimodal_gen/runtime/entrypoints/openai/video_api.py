# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


# NOTE(mick): the sampling params needs to be further adjusted
# FIXME: duplicated with the one in `image_api.py`
def _build_sampling_params_from_request(
    request_id: str, request: VideoGenerationsRequest
) -> SamplingParams:
    if request.size is None:
        width, height = None, None
    else:
        width, height = _parse_size(request.size)
    seconds = request.seconds if request.seconds is not None else 4
    # Prefer user-provided fps/num_frames from request; fallback to defaults
    fps_default = 24
    fps = request.fps if request.fps is not None else fps_default
    # If user provides num_frames, use it directly; otherwise derive from seconds * fps
    derived_num_frames = fps * seconds
    num_frames = (
        request.num_frames if request.num_frames is not None else derived_num_frames
    )
    server_args = get_global_server_args()
    sampling_kwargs = {
        "request_id": request_id,
        "prompt": request.prompt,
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "image_path": request.input_reference,
        "save_output": True,
        "output_file_name": request_id,
        "seed": request.seed,
        "generator_device": request.generator_device,
    }
    if request.num_inference_steps is not None:
        sampling_kwargs["num_inference_steps"] = request.num_inference_steps
    if request.guidance_scale is not None:
        sampling_kwargs["guidance_scale"] = request.guidance_scale
    if request.guidance_scale_2 is not None:
        sampling_kwargs["guidance_scale_2"] = request.guidance_scale_2
    if request.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = request.negative_prompt
    if request.enable_teacache is not None:
        sampling_kwargs["enable_teacache"] = request.enable_teacache
    sampling_params = SamplingParams.from_user_sampling_params_args(
        model_path=server_args.model_path,
        server_args=server_args,
        **sampling_kwargs,
    )

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
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        _, result = await process_generation_batch(async_scheduler_client, batch)
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
        }
        if result.peak_memory_mb and result.peak_memory_mb > 0:
            update_fields["peak_memory_mb"] = result.peak_memory_mb
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.error(f"{e}")
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        if input_reference is None and reference_url is None:
            raise HTTPException(
                status_code=400,
                detail="input_reference file or reference_url is required",
            )
        image_list = merge_image_input_list(input_reference, reference_url)
        # Save first input image
        image = image_list[0]
        uploads_dir = os.path.join("outputs", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        filename = image.filename if hasattr(image, "filename") else f"url_image"
        input_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
        try:
            input_path = await save_image_to_path(image, input_path)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        # Parse extra_body JSON (if provided in multipart form) to get fps/num_frames overrides
        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = json.loads(extra_body)
            except Exception:
                extra_from_form = {}

        fps_val = fps if fps is not None else extra_from_form.get("fps")
        num_frames_val = (
            num_frames if num_frames is not None else extra_from_form.get("num_frames")
        )

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            model=model,
            seconds=seconds if seconds is not None else 4,
            size=size,
            fps=fps_val,
            num_frames=num_frames_val,
            seed=seed,
            generator_device=generator_device,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            # If client uses extra_body, merge it into the top-level payload
            payload: Dict[str, Any] = dict(body or {})
            extra = payload.pop("extra_body", None)
            if isinstance(extra, dict):
                # Shallow-merge: only keys like fps/num_frames are expected
                payload.update(extra)
            # openai may turn extra_body to extra_json
            extra_json = payload.pop("extra_json", None)
            if isinstance(extra_json, dict):
                payload.update(extra_json)
            # for not multipart/form-data type
            if payload.get("reference_url"):
                image_list = merge_image_input_list(payload.get("reference_url"))
                # Save first input image
                image = image_list[0]
                uploads_dir = os.path.join("outputs", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                filename = (
                    image.filename if hasattr(image, "filename") else f"url_image"
                )
                input_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
                try:
                    input_path = await save_image_to_path(image, input_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    )
                payload["input_reference"] = input_path
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.debug(f"Server received from create_video endpoint: req={req}")

    sampling_params = _build_sampling_params_from_request(request_id, req)
    job = _video_job_from_sampling(request_id, req, sampling_params)
    await VIDEO_STORE.upsert(request_id, job)

    # Build Req for scheduler
    batch = prepare_request(
        server_args=get_global_server_args(),
        sampling_params=sampling_params,
    )
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(_dispatch_job_async(request_id, batch))
    return VideoResponse(**job)


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
    jobs = await VIDEO_STORE.list_values()

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
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(**job)


# TODO: support aborting a job.
@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.pop(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    # Mark as deleted in response semantics
    job["status"] = "deleted"
    return VideoResponse(**job)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
