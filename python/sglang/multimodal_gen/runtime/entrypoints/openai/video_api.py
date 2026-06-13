# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
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
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    add_common_data_to_response,
    build_sampling_params,
    flatten_extra_params,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])

_VIDEO_EXTENSIONS = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def _extra_value(request: VideoGenerationsRequest, name: str) -> Any:
    return (request.model_extra or {}).get(name)


def _request_value(request: VideoGenerationsRequest, name: str) -> Any:
    value = getattr(request, name, None)
    if value is not None:
        return value
    return _extra_value(request, name)


def _parse_form_extra_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _is_probably_video_source(source: Any) -> bool:
    content_type = (getattr(source, "content_type", "") or "").lower()
    if content_type.startswith("video/"):
        return True

    if isinstance(source, str):
        if source.lower().startswith("data:video"):
            return True
        source_name = source
    else:
        source_name = getattr(source, "filename", None)

    if not source_name:
        return False
    source_name = str(source_name).split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(source_name)[1].lower() in _VIDEO_EXTENSIONS


def _is_cosmos3_server(server_args) -> bool:
    pipeline_config = getattr(server_args, "pipeline_config", None)
    values = (
        getattr(server_args, "model_path", None),
        getattr(server_args, "pipeline_class_name", None),
        type(pipeline_config).__name__ if pipeline_config is not None else None,
    )
    return any("cosmos3" in str(value).lower() for value in values if value)


def _normalize_optional_string(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _coerce_optional_int_list(value: Any) -> list[int] | None:
    value = _parse_form_extra_value(value)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _resolve_video_path(req: VideoGenerationsRequest) -> str | None:
    video_path = _request_value(req, "video_path") or _request_value(req, "video_url")
    if video_path:
        return str(video_path)

    input_reference = _request_value(req, "input_reference")
    if _is_probably_video_source(input_reference):
        return str(input_reference)

    reference_url = _request_value(req, "reference_url")
    if _is_probably_video_source(reference_url):
        return str(reference_url)

    return None


def _resolve_image_path(
    req: VideoGenerationsRequest, video_path: str | None
) -> str | None:
    image_path = _request_value(req, "input_reference")
    if video_path and image_path == video_path:
        return None
    if _is_probably_video_source(image_path):
        return None
    return image_path


def _resolve_sound_duration(
    req: VideoGenerationsRequest, *, num_frames: int, fps: int
) -> float | None:
    generate_sound = _request_value(req, "generate_sound")
    sound_duration = _request_value(req, "sound_duration")

    if generate_sound is False:
        return 0.0
    if sound_duration is not None:
        return float(sound_duration)
    if generate_sound is True:
        return float(num_frames) / float(fps)
    return None


def _cosmos3_sampling_param_kwargs(
    req: VideoGenerationsRequest, *, num_frames: int, fps: int
) -> Dict[str, Any]:
    """Map HTTP/API aliases to Cosmos3SamplingParams field names."""
    kwargs: Dict[str, Any] = {}

    sound_duration = _resolve_sound_duration(req, num_frames=num_frames, fps=fps)
    if sound_duration is not None:
        kwargs["sound_duration"] = sound_duration

    condition_frame_indexes = _request_value(req, "condition_frame_indexes")
    if condition_frame_indexes is None:
        condition_frame_indexes = _request_value(
            req, "condition_frame_indexes_vision"
        )
    condition_frame_indexes = _coerce_optional_int_list(condition_frame_indexes)
    if condition_frame_indexes is not None:
        kwargs["condition_frame_indexes"] = condition_frame_indexes

    for name in (
        "condition_video_keep",
        "action_mode",
        "domain_id",
        "domain_name",
        "raw_action_dim",
        "action_fps",
        "action",
        "action_view_point",
        "action_stats_path",
        "action_normalization",
    ):
        value = _parse_form_extra_value(_request_value(req, name))
        value = _normalize_optional_string(value)
        if value is not None:
            kwargs[name] = value

    return kwargs


def _build_video_sampling_params(request_id: str, request: VideoGenerationsRequest):
    """Resolve video-specific defaults (fps, seconds → num_frames) then
    delegate to the shared build_sampling_params."""
    server_args = get_global_server_args()
    seconds = request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
    fps = request.fps if request.fps is not None else DEFAULT_FPS
    num_frames = (
        request.num_frames if request.num_frames is not None else fps * seconds
    )
    num_outputs = request.num_outputs_per_prompt
    if num_outputs is None:
        num_outputs = request.n or 1
    video_path = _resolve_video_path(request)
    image_path = _resolve_image_path(request, video_path)
    cosmos3_kwargs = (
        _cosmos3_sampling_param_kwargs(request, num_frames=num_frames, fps=fps)
        if _is_cosmos3_server(server_args)
        else {}
    )

    return build_sampling_params(
        request_id,
        prompt=request.prompt,
        num_outputs_per_prompt=max(1, min(int(num_outputs), 10)),
        size=request.size,
        width=request.width,
        height=request.height,
        num_frames=num_frames,
        fps=fps,
        image_path=image_path,
        video_path=video_path,
        output_file_name=request_id,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        guidance_scale_2=request.guidance_scale_2,
        negative_prompt=request.negative_prompt,
        max_sequence_length=request.max_sequence_length,
        flow_shift=request.flow_shift,
        use_duration_template=_extra_value(request, "use_duration_template"),
        use_resolution_template=_extra_value(request, "use_resolution_template"),
        use_system_prompt=_extra_value(request, "use_system_prompt"),
        use_guardrails=_extra_value(request, "use_guardrails"),
        enable_teacache=request.enable_teacache,
        enable_frame_interpolation=request.enable_frame_interpolation,
        frame_interpolation_exp=request.frame_interpolation_exp,
        frame_interpolation_scale=request.frame_interpolation_scale,
        frame_interpolation_model_path=request.frame_interpolation_model_path,
        enable_upscaling=request.enable_upscaling,
        upscaling_model_path=request.upscaling_model_path,
        upscaling_scale=request.upscaling_scale,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        perf_dump_path=request.perf_dump_path,
        diffusers_kwargs=request.diffusers_kwargs,
        **cosmos3_kwargs,
    )


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
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _save_first_input_image(
    image_sources,
    request_id: str,
    uploads_dir: str,
    *,
    prefer_remote_source: bool = False,
) -> str | None:
    """Save the first input image from a list of sources and return its path."""
    image_list = merge_image_input_list(image_sources)
    if not image_list:
        return None
    image = image_list[0]

    os.makedirs(uploads_dir, exist_ok=True)

    filename = image.filename if hasattr(image, "filename") else "url_image"
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(
        image, target_path, prefer_remote_source=prefer_remote_source
    )


async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

        persistent_path = (
            save_file_path if not cloud_url and output_persistent else None
        )
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": persistent_path,
            "file_paths": (
                [os.path.abspath(path) for path in save_file_path_list]
                if output_persistent
                else None
            ),
            "num_outputs": len(save_file_path_list),
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.error(f"{e}")
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
    finally:
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)


# TODO: support image to video generation
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    video_reference: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    video_path: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    num_outputs_per_prompt: Optional[int] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    max_sequence_length: Optional[int] = Form(None),
    flow_shift: Optional[float] = Form(None),
    enable_teacache: Optional[bool] = Form(None),
    generate_sound: Optional[bool] = Form(None),
    sound_duration: Optional[float] = Form(None),
    condition_frame_indexes: Optional[str] = Form(None),
    condition_frame_indexes_vision: Optional[str] = Form(None),
    condition_video_keep: Optional[str] = Form(None),
    action_mode: Optional[str] = Form(None),
    domain_id: Optional[int] = Form(None),
    domain_name: Optional[str] = Form(None),
    raw_action_dim: Optional[int] = Form(None),
    action_fps: Optional[float] = Form(None),
    action: Optional[str] = Form(None),
    action_view_point: Optional[str] = Form(None),
    action_stats_path: Optional[str] = Form(None),
    action_normalization: Optional[str] = Form(None),
    enable_frame_interpolation: Optional[bool] = Form(None),
    frame_interpolation_exp: Optional[int] = Form(None),
    frame_interpolation_scale: Optional[float] = Form(None),
    frame_interpolation_model_path: Optional[str] = Form(None),
    enable_upscaling: Optional[bool] = Form(None),
    upscaling_model_path: Optional[str] = Form(None),
    upscaling_scale: Optional[int] = Form(None),
    output_quality: Optional[str] = Form(None),
    output_compression: Optional[int] = Form(None),
    output_path: Optional[str] = Form(None),
    extra_params: Optional[str] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    # Resolve input upload directory (may be a temp dir when saving is disabled)
    temp_dirs: list[str] = []
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")
        temp_dirs.append(uploads_dir)

    # Resolve output directory
    effective_output_path = server_args.output_path
    output_persistent = True
    if "multipart/form-data" not in content_type:
        # JSON body may carry a per-request output_path; checked after parsing below
        pass

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        video_input_path = None
        image_sources = merge_image_input_list(input_reference, reference_url)
        if video_reference is not None:
            video_input_path = await _save_first_input_image(
                video_reference,
                request_id,
                uploads_dir,
                prefer_remote_source=server_args.input_save_path is None,
            )
        elif video_path or video_url:
            video_input_path = video_path or video_url
        elif input_reference is not None and _is_probably_video_source(
            input_reference
        ):
            video_input_path = await _save_first_input_image(
                input_reference,
                request_id,
                uploads_dir,
                prefer_remote_source=server_args.input_save_path is None,
            )
            image_sources = merge_image_input_list(reference_url)
        elif reference_url and _is_probably_video_source(reference_url):
            video_input_path = reference_url
            image_sources = merge_image_input_list(input_reference)

        # Validate image input based on model task type
        if task_type.requires_image_input() and not image_sources:
            raise HTTPException(
                status_code=400,
                detail="input_reference or reference_url is required for image-to-video generation",
            )
        input_path = None
        if image_sources:
            try:
                input_path = await _save_first_input_image(
                    image_sources,
                    request_id,
                    uploads_dir,
                    prefer_remote_source=server_args.input_save_path is None,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to process image source: {str(e)}"
                )

        # Parse extra_body JSON (if provided in multipart form) to get fps/num_frames overrides
        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = flatten_extra_params(json.loads(extra_body))
            except Exception:
                extra_from_form = {}
        if extra_params:
            try:
                extra_from_form.update(
                    flatten_extra_params({"extra_params": json.loads(extra_params)})
                )
            except Exception:
                pass

        def form_value(name: str, value: Any) -> Any:
            selected = value if value is not None else extra_from_form.get(name)
            return _parse_form_extra_value(selected)

        raw_form = await request.form()
        for key in (
            "use_duration_template",
            "use_resolution_template",
            "use_system_prompt",
            "use_guardrails",
            "guardrails",
            "video_path",
            "video_url",
            "generate_sound",
            "sound_duration",
            "condition_frame_indexes",
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
            "action_view_point",
            "action_stats_path",
            "action_normalization",
            "condition_frame_indexes_vision",
            "condition_video_keep",
        ):
            if key in raw_form and key not in extra_from_form:
                extra_from_form[key] = _parse_form_extra_value(raw_form[key])
        flatten_extra_params(extra_from_form)

        request_field_names = set(VideoGenerationsRequest.model_fields)
        extra_request_fields = {
            key: value
            for key, value in extra_from_form.items()
            if key not in request_field_names
        }
        fps_val = form_value("fps", fps)
        num_frames_val = form_value("num_frames", num_frames)

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            video_path=form_value("video_path", video_input_path),
            video_url=form_value("video_url", video_url),
            model=form_value("model", model),
            n=form_value("n", n),
            num_outputs_per_prompt=form_value(
                "num_outputs_per_prompt", num_outputs_per_prompt
            ),
            seconds=form_value("seconds", seconds) or 4,
            size=form_value("size", size),
            fps=fps_val,
            num_frames=num_frames_val,
            seed=form_value("seed", seed),
            generator_device=form_value("generator_device", generator_device),
            negative_prompt=form_value("negative_prompt", negative_prompt),
            num_inference_steps=form_value("num_inference_steps", num_inference_steps),
            guidance_scale=form_value("guidance_scale", guidance_scale),
            max_sequence_length=form_value("max_sequence_length", max_sequence_length),
            flow_shift=form_value("flow_shift", flow_shift),
            enable_teacache=form_value("enable_teacache", enable_teacache),
            generate_sound=form_value("generate_sound", generate_sound),
            sound_duration=form_value("sound_duration", sound_duration),
            condition_frame_indexes=form_value(
                "condition_frame_indexes", condition_frame_indexes
            ),
            condition_frame_indexes_vision=form_value(
                "condition_frame_indexes_vision", condition_frame_indexes_vision
            ),
            condition_video_keep=form_value(
                "condition_video_keep", condition_video_keep
            ),
            action_mode=form_value("action_mode", action_mode),
            domain_id=form_value("domain_id", domain_id),
            domain_name=form_value("domain_name", domain_name),
            raw_action_dim=form_value("raw_action_dim", raw_action_dim),
            action_fps=form_value("action_fps", action_fps),
            action=form_value("action", action),
            action_view_point=form_value("action_view_point", action_view_point),
            action_stats_path=form_value("action_stats_path", action_stats_path),
            action_normalization=form_value(
                "action_normalization", action_normalization
            ),
            enable_frame_interpolation=form_value(
                "enable_frame_interpolation", enable_frame_interpolation
            ),
            frame_interpolation_exp=form_value(
                "frame_interpolation_exp", frame_interpolation_exp
            ),
            frame_interpolation_scale=form_value(
                "frame_interpolation_scale", frame_interpolation_scale
            ),
            frame_interpolation_model_path=form_value(
                "frame_interpolation_model_path", frame_interpolation_model_path
            ),
            enable_upscaling=form_value("enable_upscaling", enable_upscaling),
            upscaling_model_path=form_value(
                "upscaling_model_path", upscaling_model_path
            ),
            upscaling_scale=form_value("upscaling_scale", upscaling_scale),
            output_compression=form_value("output_compression", output_compression),
            output_quality=form_value("output_quality", output_quality),
            output_path=form_value("output_path", output_path),
            diffusers_kwargs=form_value("diffusers_kwargs", None),
            **extra_request_fields,
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
            if isinstance(extra, str):
                extra = json.loads(extra)
            if isinstance(extra, dict):
                payload.update(flatten_extra_params(extra))
            # openai may turn extra_body to extra_json
            extra_json = payload.pop("extra_json", None)
            if isinstance(extra_json, str):
                extra_json = json.loads(extra_json)
            if isinstance(extra_json, dict):
                payload.update(flatten_extra_params(extra_json))
            flatten_extra_params(payload)
            # Validate image input based on model task type
            if payload.get("video_url") and not payload.get("video_path"):
                payload["video_path"] = payload["video_url"]
            if _is_probably_video_source(payload.get("reference_url")):
                payload.setdefault("video_path", payload.get("reference_url"))
            if _is_probably_video_source(payload.get("input_reference")):
                payload.setdefault("video_path", payload.get("input_reference"))

            has_image_input = (
                payload.get("reference_url")
                and not _is_probably_video_source(payload.get("reference_url"))
            ) or (
                payload.get("input_reference")
                and not _is_probably_video_source(payload.get("input_reference"))
            )
            if task_type.requires_image_input() and not has_image_input:
                raise HTTPException(
                    status_code=400,
                    detail="input_reference or reference_url is required for image-to-video generation",
                )
            # for non-multipart/form-data type
            if payload.get("reference_url") and not _is_probably_video_source(
                payload.get("reference_url")
            ):
                try:
                    input_path = await _save_first_input_image(
                        payload.get("reference_url"),
                        request_id,
                        uploads_dir,
                        prefer_remote_source=server_args.input_save_path is None,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    )
                payload["input_reference"] = input_path
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # Resolve per-request output_path override
    effective_output_path = req.output_path or server_args.output_path
    if effective_output_path is None:
        output_tmp = tempfile.mkdtemp(prefix="sglang_output_")
        temp_dirs.append(output_tmp)
        effective_output_path = output_tmp
        output_persistent = False

    # Inject resolved output_path so _build_video_sampling_params picks it up
    req.output_path = effective_output_path

    logger.debug(f"Server received from create_video endpoint: req={req}")

    try:
        sampling_params = _build_video_sampling_params(request_id, req)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = _video_job_from_sampling(request_id, req, sampling_params)
    await VIDEO_STORE.upsert(request_id, job)

    # Build Req for scheduler
    trace_headers = extract_trace_headers(request.headers)
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
        external_trace_header=trace_headers,
    )
    # Add diffusers_kwargs if provided
    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
        if "max_sequence_length" in req.diffusers_kwargs:
            batch.max_sequence_length = req.diffusers_kwargs["max_sequence_length"]
        if "flow_shift" in req.diffusers_kwargs:
            batch.flow_shift = req.diffusers_kwargs["flow_shift"]
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(
        _dispatch_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        )
    )
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

    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Video has been uploaded to cloud storage. Please use the cloud URL: {job.get('url')}",
        )

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
