# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import Coroutine
from contextlib import suppress
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
    DataType,
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
_VIDEO_JOB_TASKS: dict[str, asyncio.Task[None]] = {}


def _discard_video_job_task(job_id: str, task: asyncio.Task[None]) -> None:
    if _VIDEO_JOB_TASKS.get(job_id) is task:
        del _VIDEO_JOB_TASKS[job_id]


def _start_video_job(job_id: str, job: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(job, name=f"video-job-{job_id}")
    _VIDEO_JOB_TASKS[job_id] = task
    task.add_done_callback(lambda completed: _discard_video_job_task(job_id, completed))
    return task


async def shutdown_video_jobs() -> None:
    tasks = list(_VIDEO_JOB_TASKS.values())
    _VIDEO_JOB_TASKS.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task


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


_MULTIPART_EXTRA_FORM_FIELDS = (
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
    "action_normalization",
    "condition_frame_indexes_vision",
    "condition_video_keep",
    "quality",
)


def _video_sampling_params_cls(server_args) -> type[SamplingParams]:
    """Resolve the params type selected for the current server."""

    sampling_params_cls = SamplingParams
    if server_args.pipeline_class_name:
        from sglang.multimodal_gen.registry import get_pipeline_config_classes

        config_classes = get_pipeline_config_classes(server_args.pipeline_class_name)
        if config_classes is not None:
            _, sampling_params_cls = config_classes
    if sampling_params_cls is SamplingParams:
        from sglang.multimodal_gen.registry import get_model_info

        model_info = get_model_info(
            server_args.model_path,
            backend=server_args.backend,
            model_id=server_args.model_id,
        )
        if model_info is not None:
            sampling_params_cls = model_info.sampling_param_cls
    return sampling_params_cls


def _multipart_extra_form_keys(
    sampling_params_cls: type[SamplingParams],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *VideoGenerationsRequest.model_fields,
                *_MULTIPART_EXTRA_FORM_FIELDS,
                *sorted(sampling_params_cls.video_request_extra_fields()),
            )
        )
    )


def _filter_multipart_declared_fields(
    extra_from_form: Dict[str, Any],
    sampling_params_cls: type[SamplingParams],
) -> Dict[str, Any]:
    declared = set(_multipart_extra_form_keys(sampling_params_cls))
    return {key: value for key, value in extra_from_form.items() if key in declared}


def _merge_multipart_extra_form_fields(
    raw_form: Any,
    extra_from_form: Dict[str, Any],
    sampling_params_cls: type[SamplingParams],
) -> None:
    for key in _multipart_extra_form_keys(sampling_params_cls):
        if key in raw_form and key not in extra_from_form:
            extra_from_form[key] = _parse_form_extra_value(raw_form[key])


def _multipart_video_extras(
    raw_form: Any,
    *,
    extra_body: Any,
    extra_params: Any,
    sampling_params_cls: type[SamplingParams],
) -> Dict[str, Any]:
    """Build and validate multipart extras once for request construction."""

    extra_from_form: Dict[str, Any] = {}
    if extra_body:
        try:
            extra_from_form = flatten_extra_params(json.loads(extra_body))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_body is not valid JSON"
            ) from exc
    if extra_params:
        try:
            extra_from_form.update(
                flatten_extra_params({"extra_params": json.loads(extra_params)})
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail="extra_params is not valid JSON"
            ) from exc
    _merge_multipart_extra_form_fields(
        raw_form,
        extra_from_form,
        sampling_params_cls,
    )
    flatten_extra_params(extra_from_form)
    return _filter_multipart_declared_fields(extra_from_form, sampling_params_cls)


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
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config

    return isinstance(server_args.pipeline_config, Cosmos3Config)


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
        condition_frame_indexes = _request_value(req, "condition_frame_indexes_vision")
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
    num_frames = request.num_frames if request.num_frames is not None else fps * seconds
    num_outputs = request.num_outputs_per_prompt
    if num_outputs is None:
        num_outputs = request.n or 1
    video_path = _resolve_video_path(request)
    image_path = _resolve_image_path(request, video_path)
    cosmos3_kwargs = {}
    if _is_cosmos3_server(server_args):
        cosmos3_kwargs = _cosmos3_sampling_param_kwargs(
            request, num_frames=num_frames, fps=fps
        )
        if server_args.pipeline_config.action_stats_path is not None:
            cosmos3_kwargs["action_stats_path"] = (
                server_args.pipeline_config.action_stats_path
            )

    kwargs = {
        "prompt": request.prompt,
        "num_outputs_per_prompt": max(1, min(int(num_outputs), 10)),
        "size": request.size,
        "width": request.width,
        "height": request.height,
        "num_frames": num_frames,
        "fps": fps,
        "image_path": image_path,
        "video_path": video_path,
        "output_file_name": request_id,
        "seed": request.seed,
        "generator_device": request.generator_device,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "guidance_scale_2": request.guidance_scale_2,
        "true_cfg_scale": request.true_cfg_scale,
        "negative_prompt": request.negative_prompt,
        "max_sequence_length": request.max_sequence_length,
        "flow_shift": request.flow_shift,
        "use_duration_template": _extra_value(request, "use_duration_template"),
        "use_resolution_template": _extra_value(request, "use_resolution_template"),
        "use_system_prompt": _extra_value(request, "use_system_prompt"),
        "use_guardrails": _extra_value(request, "use_guardrails"),
        "enable_teacache": request.enable_teacache,
        "enable_frame_interpolation": request.enable_frame_interpolation,
        "frame_interpolation_exp": request.frame_interpolation_exp,
        "frame_interpolation_scale": request.frame_interpolation_scale,
        "frame_interpolation_model_path": request.frame_interpolation_model_path,
        "enable_upscaling": request.enable_upscaling,
        "upscaling_model_path": request.upscaling_model_path,
        "upscaling_scale": request.upscaling_scale,
        "output_path": request.output_path,
        "quality": _extra_value(request, "quality"),
        "output_compression": request.output_compression,
        "output_quality": request.output_quality,
        "perf_dump_path": request.perf_dump_path,
        "profile": request.profile,
        "num_profiled_timesteps": request.num_profiled_timesteps,
        "profile_all_stages": request.profile_all_stages,
        "diffusers_kwargs": request.diffusers_kwargs,
        **cosmos3_kwargs,
    }

    sampling_params_cls = _video_sampling_params_cls(server_args)
    kwargs = sampling_params_cls.lower_video_request_kwargs(request, kwargs)
    sampling_params = build_sampling_params(request_id, **kwargs)
    if (
        isinstance(sampling_params, SamplingParams)
        and sampling_params.data_type == DataType.ACTION
    ):
        raise ValueError(
            "Action-producing policy and inverse-dynamics requests use "
            "/v1/actions/generations; /v1/videos is reserved for visual outputs"
        )
    return sampling_params


# extract metadata which http_server needs to know
def _video_job_from_sampling(
    request_id: str,
    req: VideoGenerationsRequest,
    sampling: SamplingParams,
    served_model_name: str,
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or served_model_name,
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
    scheduler_batches: list[Req] | None = None,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client,
            batch,
            scheduler_batches=scheduler_batches,
        )
        save_file_path = save_file_path_list[0]
        try:
            final_media_fields = await asyncio.to_thread(
                batch.sampling_params.validate_video_final_outputs,
                save_file_path_list,
                batch,
            )
        except Exception:
            for output_path in save_file_path_list:
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Failed to remove rejected video output %s",
                        output_path,
                        exc_info=True,
                    )
            raise

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
        update_fields.update(final_media_fields)
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        logger.exception("Video job %s failed", job_id)
        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": str(e)},
                "url": None,
                "file_path": None,
                "file_paths": None,
                "num_outputs": None,
            },
        )
    finally:
        try:
            await asyncio.to_thread(
                batch.sampling_params.cleanup_video_request,
                batch,
            )
        except Exception:
            logger.warning(
                "Failed to clean up model-owned video request resources",
                exc_info=True,
            )
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
    guidance_scale_2: Optional[float] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    max_sequence_length: Optional[int] = Form(None),
    flow_shift: Optional[float] = Form(None),
    enable_teacache: Optional[bool] = Form(None),
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
    is_multipart = "multipart/form-data" in content_type
    raw_form: Any = None
    extra_from_form: Dict[str, Any] = {}

    # Parse model-specific multipart metadata before creating request-owned
    # directories or saving uploads, so malformed JSON leaves no resources.
    if is_multipart:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        raw_form = await request.form()
        extra_from_form = _multipart_video_extras(
            raw_form,
            extra_body=extra_body,
            extra_params=extra_params,
            sampling_params_cls=_video_sampling_params_cls(server_args),
        )

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
    if not is_multipart:
        # JSON body may carry a per-request output_path; checked after parsing below
        pass

    if is_multipart:
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
        elif input_reference is not None and _is_probably_video_source(input_reference):
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

        def form_value(name: str, value: Any) -> Any:
            selected = value if value is not None else extra_from_form.get(name)
            return _parse_form_extra_value(selected)

        def form_text_value(name: str, value: Any) -> Any:
            """Resolve a text field without JSON-decoding it.

            Some models take a serialized JSON object as the prompt text, which
            ``_parse_form_extra_value`` would turn back into a dict and fail
            request validation.
            """
            return value if value is not None else extra_from_form.get(name)

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
            negative_prompt=form_text_value("negative_prompt", negative_prompt),
            num_inference_steps=form_value("num_inference_steps", num_inference_steps),
            guidance_scale=form_value("guidance_scale", guidance_scale),
            guidance_scale_2=form_value("guidance_scale_2", guidance_scale_2),
            true_cfg_scale=form_value("true_cfg_scale", true_cfg_scale),
            max_sequence_length=form_value("max_sequence_length", max_sequence_length),
            flow_shift=form_value("flow_shift", flow_shift),
            enable_teacache=form_value("enable_teacache", enable_teacache),
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
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

    batch: Req | None = None
    scheduler_batches: list[Req] | None = None
    try:
        # Build Req for scheduler.
        trace_headers = extract_trace_headers(request.headers)
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
            external_trace_header=trace_headers,
        )
        # Add diffusers_kwargs if provided.
        if req.diffusers_kwargs:
            batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
            if "max_sequence_length" in req.diffusers_kwargs:
                batch.max_sequence_length = req.diffusers_kwargs["max_sequence_length"]
            if "flow_shift" in req.diffusers_kwargs:
                batch.flow_shift = req.diffusers_kwargs["flow_shift"]
        await asyncio.to_thread(
            sampling_params.prepare_video_request_for_queue,
            batch,
        )
        scheduler_batches = sampling_params.expand_video_request_outputs_for_queue(
            batch
        )
        job = _video_job_from_sampling(
            request_id,
            req,
            sampling_params,
            server_args.served_model_name,
        )
        job.update(sampling_params.project_video_queued_job_fields(batch))
        await VIDEO_STORE.upsert(request_id, job)
    except Exception as e:
        if batch is not None:
            try:
                await asyncio.to_thread(sampling_params.cleanup_video_request, batch)
            except Exception:
                logger.warning(
                    "Failed to clean up rejected video request resources",
                    exc_info=True,
                )
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)
        if isinstance(e, (TypeError, ValueError)):
            raise HTTPException(status_code=400, detail=str(e)) from e
        raise

    assert batch is not None
    # Enqueue the job asynchronously and return immediately
    _start_video_job(
        request_id,
        _dispatch_job_async(
            request_id,
            batch,
            scheduler_batches=scheduler_batches,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
        ),
    )
    return VideoResponse(**job)


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    jobs = await VIDEO_STORE.list_page(after=after, limit=limit, order=order)
    items = [VideoResponse(**job) for job in jobs]
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


def _select_video_variant_path(job: dict, variant: str | None) -> str | None:
    file_paths = job.get("file_paths")
    if file_paths:
        try:
            variant_index = 0 if variant is None else int(variant)
        except (TypeError, ValueError):
            return None
        if 0 <= variant_index < len(file_paths):
            return file_paths[variant_index]
        return None
    if variant not in (None, "0", 0):
        return None
    return job.get("file_path")


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

    file_path = _select_video_variant_path(job, variant)
    if job.get("status") not in {"completed", "failed"}:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail=f"Video variant {variant} not found"
        )

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
