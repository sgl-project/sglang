# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import contextlib
import json
import os
import time
from typing import Any, List, Optional

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
from fastapi.responses import FileResponse, StreamingResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import IMAGE_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    build_sampling_params,
    choose_output_image_ext,
    flatten_extra_params,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.srt.observability.trace import extract_trace_headers

router = APIRouter(prefix="/v1/images", tags=["images"])


def _get_extra_field(request, field_name):
    """Get a field from model_extra, with fallback to nested extra_body dict."""
    extra = request.model_extra or {}
    value = extra.get(field_name)
    if value is not None:
        return value
    if field_name == "use_guardrails" and extra.get("guardrails") is not None:
        return extra["guardrails"]

    for container_name in ("extra_body", "extra_json", "extra_args", "extra_params"):
        value = _parse_extra_container(extra.get(container_name)).get(field_name)
        if value is not None:
            return value

    return value


def _get_request_field_or_extra(request, field_name):
    value = getattr(request, field_name, None)
    if value is not None:
        return value
    return _get_extra_field(request, field_name)


def _parse_extra_container(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    if isinstance(value, dict):
        return flatten_extra_params(dict(value))
    return {}


def _read_b64_for_paths(paths: list[str]) -> list[str]:
    """Read and base64-encode each file. Must be called before cloud upload deletes them."""
    result = []
    for path in paths:
        with open(path, "rb") as f:
            result.append(base64.b64encode(f.read()).decode("utf-8"))
    return result


async def _upload_and_cleanup_images(paths: list[str]) -> list[str | None]:
    return await asyncio.gather(
        *(cloud_storage.upload_and_cleanup(path) for path in paths)
    )


def _fallback_image_urls(
    request_id: str, num_outputs: int, is_persistent: bool
) -> list[str] | None:
    if not is_persistent:
        return None
    if num_outputs <= 1:
        return [f"/v1/images/{request_id}/content"]
    return [
        f"/v1/images/{request_id}/content?variant={idx}" for idx in range(num_outputs)
    ]


def _select_image_variant_path(item: dict, variant: str | None) -> str | None:
    file_paths = item.get("file_paths")
    if file_paths:
        variant_idx = _image_variant_index(variant)
        if variant_idx is None:
            return None
        if variant_idx < 0 or variant_idx >= len(file_paths):
            return None
        return file_paths[variant_idx]

    if variant not in (None, "0", 0):
        return None
    return item.get("file_path")


def _image_variant_index(variant: str | None) -> int | None:
    try:
        return 0 if variant is None else int(variant)
    except (TypeError, ValueError):
        return None


def _select_image_variant_cloud_url(item: dict, variant: str | None) -> str | None:
    variant_idx = _image_variant_index(variant)
    if variant_idx is None:
        return None

    urls = item.get("urls")
    if urls and 0 <= variant_idx < len(urls):
        return urls[variant_idx]
    if variant_idx == 0:
        return item.get("url")
    return None


def _raise_if_image_variant_not_found(item: dict, variant: str | None) -> None:
    file_paths = item.get("file_paths")
    if not file_paths:
        return

    variant_idx = _image_variant_index(variant)
    if variant_idx is None or variant_idx < 0 or variant_idx >= len(file_paths):
        raise HTTPException(
            status_code=404,
            detail=f"Image variant {variant} not found",
        )


def _build_image_response_kwargs(
    save_file_path_list: list[str],
    resp_format: str,
    prompt: str,
    request_id: str,
    result: OutputBatch,
    *,
    b64_list: list[str] | None = None,
    cloud_url: str | None = None,
    cloud_urls: list[str | None] | None = None,
    fallback_url: str | None = None,
    fallback_urls: list[str] | None = None,
    is_persistent: bool = True,
) -> dict:
    """Build ImageResponse data list.

    For b64_json: uses pre-read b64_list (call _read_b64_for_paths first).
    For url: uses cloud_url or fallback_url.
    file_path is omitted when is_persistent=False to avoid exposing stale temp paths.
    """
    ret = None
    if resp_format == "b64_json":
        if not b64_list:
            raise ValueError("b64_list required for b64_json response_format")
        data = [
            ImageResponseData(
                b64_json=b64,
                revised_prompt=prompt,
                file_path=os.path.abspath(path) if is_persistent else None,
            )
            for b64, path in zip(b64_list, save_file_path_list)
        ]
        ret = {"data": data}
    elif resp_format == "url":
        if cloud_urls is None and cloud_url is not None:
            cloud_urls = [cloud_url]
        if fallback_urls is None and fallback_url is not None:
            fallback_urls = [fallback_url]

        data = []
        for idx, path in enumerate(save_file_path_list):
            url = None
            if cloud_urls is not None and idx < len(cloud_urls):
                url = cloud_urls[idx]
            if not url and fallback_urls is not None and idx < len(fallback_urls):
                url = fallback_urls[idx]
            if not url:
                break
            data.append(
                ImageResponseData(
                    url=url,
                    revised_prompt=prompt,
                    file_path=os.path.abspath(path) if is_persistent else None,
                )
            )

        if len(data) != len(save_file_path_list):
            raise HTTPException(
                status_code=400,
                detail="response_format='url' requires cloud storage to be configured.",
            )
        ret = {"data": data}
    else:
        raise HTTPException(
            status_code=400, detail=f"response_format={resp_format} is not supported"
        )

    ret = add_common_data_to_response(ret, request_id=request_id, result=result)

    return ret


async def _stream_image_response(response: ImageResponse):
    for idx, item in enumerate(response.data):
        payload = {
            "type": "image_generation.partial_image",
            "partial_image_index": idx,
        }
        if item.b64_json is not None:
            payload["b64_json"] = item.b64_json
        if item.url is not None:
            payload["url"] = item.url
        yield f"data: {json.dumps(payload)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/generations", response_model=ImageResponse)
async def generations(
    request: ImageGenerationsRequest,
    raw_request: Request,
):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    is_cosmos3 = "cosmos3" in (server_args.model_path or "").lower()
    ext = (
        "png"
        if is_cosmos3 and request.output_format is None
        else choose_output_image_ext(request.output_format, request.background)
    )

    with temp_dir_if_disabled(server_args.output_path) as output_dir:
        sampling = build_sampling_params(
            request_id,
            prompt=request.prompt,
            size=request.size,
            width=request.width,
            height=request.height,
            num_outputs_per_prompt=max(1, min(int(request.n or 1), 10)),
            output_file_name=f"{request_id}.{ext}",
            output_path=output_dir,
            num_frames=1,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            true_cfg_scale=request.true_cfg_scale,
            negative_prompt=request.negative_prompt,
            max_sequence_length=(
                request.max_sequence_length
                if request.max_sequence_length is not None
                else _get_extra_field(request, "max_sequence_length")
            ),
            flow_shift=(
                request.flow_shift
                if request.flow_shift is not None
                else _get_extra_field(request, "flow_shift")
            ),
            use_duration_template=_get_extra_field(request, "use_duration_template"),
            use_resolution_template=_get_extra_field(
                request, "use_resolution_template"
            ),
            use_system_prompt=_get_extra_field(request, "use_system_prompt"),
            use_guardrails=_get_extra_field(request, "use_guardrails"),
            enable_teacache=request.enable_teacache,
            output_compression=request.output_compression,
            output_quality=request.output_quality,
            diffusers_kwargs=request.diffusers_kwargs,
            enable_upscaling=request.enable_upscaling,
            upscaling_model_path=request.upscaling_model_path,
            upscaling_scale=request.upscaling_scale,
            perf_dump_path=request.perf_dump_path,
            use_pe=_get_extra_field(request, "use_pe"),
            preset=_get_extra_field(request, "preset"),
            progressive_mode=_get_request_field_or_extra(request, "progressive_mode"),
            progressive_levels=_get_request_field_or_extra(
                request, "progressive_levels"
            ),
            progressive_delta=_get_request_field_or_extra(request, "progressive_delta"),
        )
        trace_headers = extract_trace_headers(raw_request.headers)
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling,
            external_trace_header=trace_headers,
        )
        # Add diffusers_kwargs if provided
        if request.diffusers_kwargs:
            batch.extra["diffusers_kwargs"] = request.diffusers_kwargs

        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]
        resp_format = (request.response_format or "b64_json").lower()
        if (
            is_cosmos3
            and "response_format" not in request.model_fields_set
            and request.response_format == "url"
        ):
            resp_format = "b64_json"

        # read b64 before cloud upload may delete the local file
        b64_list = (
            _read_b64_for_paths(save_file_path_list)
            if resp_format == "b64_json"
            else None
        )

        is_persistent = server_args.output_path is not None
        cloud_urls = await _upload_and_cleanup_images(save_file_path_list)
        cloud_url = cloud_urls[0] if cloud_urls else None
        fallback_urls = _fallback_image_urls(
            request_id, len(save_file_path_list), is_persistent
        )
        await IMAGE_STORE.upsert(
            request_id,
            {
                "id": request_id,
                "created_at": int(time.time()),
                "file_path": None if cloud_url or not is_persistent else save_file_path,
                "file_paths": (
                    None
                    if not is_persistent
                    else [
                        None if url else path
                        for path, url in zip(save_file_path_list, cloud_urls)
                    ]
                ),
                "url": cloud_url,
                "urls": cloud_urls,
                "num_outputs": len(save_file_path_list),
            },
        )

        response_kwargs = _build_image_response_kwargs(
            save_file_path_list,
            resp_format,
            request.prompt,
            request_id,
            result,
            b64_list=b64_list,
            cloud_urls=cloud_urls,
            fallback_urls=fallback_urls,
            is_persistent=is_persistent,
        )

    response = ImageResponse(**response_kwargs)
    if request.stream:
        return StreamingResponse(
            _stream_image_response(response),
            media_type="text/event-stream",
        )
    return response


@router.post("/edits", response_model=ImageResponse)
async def edits(
    raw_request: Request,
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    url: Optional[List[str]] = Form(None),
    url_array: Optional[List[str]] = Form(None, alias="url[]"),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    seed: Optional[int] = Form(None),
    generator_device: Optional[str] = Form("cuda"),
    user: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    output_quality: Optional[str] = Form("default"),
    output_compression: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    enable_upscaling: Optional[bool] = Form(False),
    upscaling_model_path: Optional[str] = Form(None),
    upscaling_scale: Optional[int] = Form(4),
    num_frames: int = Form(1),
):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    # Resolve images from either `image` or `image[]` (OpenAI SDK sends `image[]` when list is provided)
    images = image or image_array
    urls = url or url_array

    if (not images or len(images) == 0) and (not urls or len(urls) == 0):
        raise HTTPException(
            status_code=422, detail="Field 'image' or 'url' is required"
        )

    image_list = merge_image_input_list(images, urls)

    with contextlib.ExitStack() as stack:
        uploads_dir = stack.enter_context(
            temp_dir_if_disabled(server_args.input_save_path)
        )
        output_dir = stack.enter_context(temp_dir_if_disabled(server_args.output_path))

        input_paths = []
        try:
            for idx, img in enumerate(image_list):
                filename = img.filename if hasattr(img, "filename") else f"image_{idx}"
                input_path = await save_image_to_path(
                    img,
                    os.path.join(uploads_dir, f"{request_id}_{idx}_{filename}"),
                    prefer_remote_source=server_args.input_save_path is None,
                )
                input_paths.append(input_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image source: {str(e)}",
            )

        ext = choose_output_image_ext(output_format, background)
        sampling = build_sampling_params(
            request_id,
            prompt=prompt,
            size=size,
            num_outputs_per_prompt=max(1, min(int(n or 1), 10)),
            output_file_name=f"{request_id}.{ext}",
            output_path=output_dir,
            image_path=input_paths,
            seed=seed,
            generator_device=generator_device,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
            num_frames=num_frames,
            output_compression=output_compression,
            output_quality=output_quality,
            enable_upscaling=enable_upscaling,
            upscaling_model_path=upscaling_model_path,
            upscaling_scale=upscaling_scale,
        )
        trace_headers = extract_trace_headers(raw_request.headers)
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling,
            external_trace_header=trace_headers,
        )
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]
        resp_format = (response_format or "b64_json").lower()

        # read b64 before cloud upload may delete the local file
        b64_list = (
            _read_b64_for_paths(save_file_path_list)
            if resp_format == "b64_json"
            else None
        )

        is_persistent = server_args.output_path is not None
        is_input_persistent = server_args.input_save_path is not None
        cloud_urls = await _upload_and_cleanup_images(save_file_path_list)
        cloud_url = cloud_urls[0] if cloud_urls else None
        fallback_urls = _fallback_image_urls(
            request_id, len(save_file_path_list), is_persistent
        )
        await IMAGE_STORE.upsert(
            request_id,
            {
                "id": request_id,
                "created_at": int(time.time()),
                "file_path": None if cloud_url or not is_persistent else save_file_path,
                "file_paths": (
                    None
                    if not is_persistent
                    else [
                        None if url else path
                        for path, url in zip(save_file_path_list, cloud_urls)
                    ]
                ),
                "url": cloud_url,
                "urls": cloud_urls,
                "input_image_paths": input_paths if is_input_persistent else None,
                "num_input_images": len(input_paths),
                "num_outputs": len(save_file_path_list),
            },
        )

        response_kwargs = _build_image_response_kwargs(
            save_file_path_list,
            resp_format,
            prompt,
            request_id,
            result,
            b64_list=b64_list,
            cloud_urls=cloud_urls,
            fallback_urls=fallback_urls,
            is_persistent=is_persistent,
        )

    return ImageResponse(**response_kwargs)


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

    _raise_if_image_variant_not_found(item, variant)
    file_path = _select_image_variant_path(item, variant)
    if not file_path:
        cloud_url = _select_image_variant_cloud_url(item, variant)
    else:
        cloud_url = None
    if not file_path and cloud_url:
        raise HTTPException(
            status_code=400,
            detail=f"Image has been uploaded to cloud storage. Please use the cloud URL: {cloud_url}",
        )

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail="Image was not persisted on disk (output_path is disabled). Use b64_json response_format or configure cloud storage.",
        )
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image is still being generated")

    ext = os.path.splitext(file_path)[1].lower()
    media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"

    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
