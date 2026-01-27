# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import base64
import os
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import IMAGE_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    add_common_data_to_response,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1/images", tags=["images"])
logger = init_logger(__name__)


def _choose_ext(output_format: Optional[str], background: Optional[str]) -> str:
    # Normalize and choose extension
    fmt = (output_format or "").lower()
    if fmt in {"png", "webp", "jpeg", "jpg"}:
        return "jpg" if fmt == "jpeg" else fmt
    # If transparency requested, prefer png
    if (background or "auto").lower() == "transparent":
        return "png"
    # Default
    return "jpg"


def _build_sampling_params_from_request(
    request_id: str,
    prompt: str,
    n: int,
    size: Optional[str],
    output_format: Optional[str],
    background: Optional[str],
    image_path: Optional[list[str]] = None,
    seed: Optional[int] = None,
    generator_device: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    true_cfg_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    enable_teacache: Optional[bool] = None,
    num_frames: int = 1,
) -> SamplingParams:
    if size is None:
        width, height = None, None
    else:
        width, height = _parse_size(size)
    ext = _choose_ext(output_format, background)

    server_args = get_global_server_args()
    sampling_params = SamplingParams.from_user_sampling_params_args(
        model_path=server_args.model_path,
        request_id=request_id,
        prompt=prompt,
        image_path=image_path,
        num_frames=num_frames,
        width=width,
        height=height,
        num_outputs_per_prompt=max(1, min(int(n or 1), 10)),
        save_output=True,
        server_args=server_args,
        output_file_name=f"{request_id}.{ext}",
        seed=seed,
        generator_device=generator_device,
        num_inference_steps=num_inference_steps,
        enable_teacache=enable_teacache,
        **({"guidance_scale": guidance_scale} if guidance_scale is not None else {}),
        **({"negative_prompt": negative_prompt} if negative_prompt is not None else {}),
        **({"true_cfg_scale": true_cfg_scale} if true_cfg_scale is not None else {}),
    )

    if num_inference_steps is not None:
        sampling_params.num_inference_steps = num_inference_steps
    if guidance_scale is not None:
        sampling_params.guidance_scale = guidance_scale
    if seed is not None:
        sampling_params.seed = seed

    return sampling_params


@router.post("/generations", response_model=ImageResponse)
async def generations(
    request: ImageGenerationsRequest,
):

    request_id = generate_request_id()
    sampling = _build_sampling_params_from_request(
        request_id=request_id,
        prompt=request.prompt,
        n=request.n or 1,
        size=request.size,
        output_format=request.output_format,
        background=request.background,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
    )
    batch = prepare_request(
        server_args=get_global_server_args(),
        sampling_params=sampling,
    )
    # Add diffusers_kwargs if provided
    if request.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = request.diffusers_kwargs

    # Run synchronously for images and save to disk
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    save_file_path = save_file_path_list[0]

    resp_format = (request.response_format or "b64_json").lower()
    b64_data = None

    # 1. Read content first if needed (while file exists)
    if resp_format == "b64_json":
        with open(save_file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

    # 2. Upload and Delete local file
    cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

    # 3. Update Database
    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": None if cloud_url else save_file_path,
            "url": cloud_url,
        },
    )

    # 4. Return Response
    if resp_format == "b64_json":
        response_kwargs = {
            "data": [
                ImageResponseData(
                    b64_json=b64_data,
                    revised_prompt=request.prompt,
                )
            ]
        }
    elif resp_format == "url":
        if not cloud_url:
            raise HTTPException(
                status_code=400,
                detail="response_format='url' requires cloud storage to be configured.",
            )
        response_kwargs = {
            "data": [
                ImageResponseData(
                    url=cloud_url,
                    revised_prompt=request.prompt,
                    file_path=os.path.abspath(save_file_path),
                )
            ],
        }
    else:
        # Return error, not supported
        raise HTTPException(
            status_code=400, detail=f"response_format={resp_format} is not supported"
        )

    response_kwargs = add_common_data_to_response(
        response_kwargs, request_id=request_id, result=result
    )
    return ImageResponse(**response_kwargs)


@router.post("/edits", response_model=ImageResponse)
async def edits(
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
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    user: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    num_frames: int = Form(1),
):
    request_id = generate_request_id()
    # Resolve images from either `image` or `image[]` (OpenAI SDK sends `image[]` when list is provided)
    images = image or image_array
    urls = url or url_array

    if (not images or len(images) == 0) and (not urls or len(urls) == 0):
        raise HTTPException(
            status_code=422, detail="Field 'image' or 'url' is required"
        )

    # Save all input images; additional images beyond the first are saved for potential future use
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    image_list = merge_image_input_list(images, urls)

    input_paths = []
    try:
        for idx, img in enumerate(image_list):
            filename = img.filename if hasattr(img, "filename") else f"image_{idx}"
            input_path = await save_image_to_path(
                img, os.path.join(uploads_dir, f"{request_id}_{idx}_{filename}")
            )
            input_paths.append(input_path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process image source: {str(e)}"
        )

    sampling = _build_sampling_params_from_request(
        request_id=request_id,
        prompt=prompt,
        n=n or 1,
        size=size,
        output_format=output_format,
        background=background,
        image_path=input_paths,
        seed=seed,
        generator_device=generator_device,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        enable_teacache=enable_teacache,
        num_frames=num_frames,
    )
    batch = prepare_request(
        server_args=get_global_server_args(),
        sampling_params=sampling,
    )

    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    save_file_path = save_file_path_list[0]

    resp_format = (response_format or "b64_json").lower()
    b64_data = None

    # 1. Read content first if needed (while file exists)
    if resp_format == "b64_json":
        with open(save_file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

    # 2. Upload and Delete local file
    cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

    # 3. Update Database
    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": None if cloud_url else save_file_path,
            "url": cloud_url,
            "input_image_paths": input_paths,  # Store all input image paths
            "num_input_images": len(input_paths),
        },
    )

    # 4. Return Response
    if (response_format or "b64_json").lower() == "b64_json":
        response_kwargs = {"data": []}
        for path in save_file_path_list:
            if path == save_file_path and b64_data is not None:
                b64 = b64_data
            else:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            response_kwargs["data"].append(
                ImageResponseData(
                    b64_json=b64,
                    revised_prompt=prompt,
                    file_path=os.path.abspath(path),
                )
            )
        if result.peak_memory_mb and result.peak_memory_mb > 0:
            response_kwargs["peak_memory_mb"] = result.peak_memory_mb
    else:
        response_kwargs = {
            "data": [
                ImageResponseData(
                    url=cloud_url if cloud_url else f"/v1/images/{request_id}/content",
                    revised_prompt=prompt,
                    file_path=os.path.abspath(save_file_path),
                )
            ],
        }

    response_kwargs = add_common_data_to_response(
        response_kwargs, request_id=request_id, result=result
    )
    return ImageResponse(**response_kwargs)


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

    if item.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Image has been uploaded to cloud storage. Please use the cloud URL: {item.get('url')}",
        )

    file_path = item.get("file_path")
    if not file_path or not os.path.exists(file_path):
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
