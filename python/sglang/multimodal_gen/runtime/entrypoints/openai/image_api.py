# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import base64
import os
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse

from sglang.multimodal_gen.configs.sample.base import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import IMAGE_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    _save_upload_to_path,
    post_process_sample,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.scheduler_client import scheduler_client
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
    image_path: Optional[str] = None,
) -> SamplingParams:
    width, height = _parse_size(size)
    ext = _choose_ext(output_format, background)

    server_args = get_global_server_args()
    sampling_params = SamplingParams.from_pretrained(server_args.model_path)

    # Build user params
    user_params = SamplingParams(
        request_id=request_id,
        prompt=prompt,
        image_path=image_path,
        num_frames=1,  # image
        width=width,
        height=height,
        num_outputs_per_prompt=max(1, min(int(n or 1), 10)),
        save_output=True,
    )

    # Let SamplingParams auto-generate a file name, then force desired extension
    sampling_params = sampling_params.from_user_sampling_params(user_params)
    if not sampling_params.output_file_name:
        sampling_params.output_file_name = request_id
    if not sampling_params.output_file_name.endswith(f".{ext}"):
        # strip any existing extension and apply desired one
        base = sampling_params.output_file_name.rsplit(".", 1)[0]
        sampling_params.output_file_name = f"{base}.{ext}"

    sampling_params.log(server_args)
    return sampling_params


def _build_req_from_sampling(s: SamplingParams) -> Req:
    return Req(
        request_id=s.request_id,
        data_type=s.data_type,
        prompt=s.prompt,
        image_path=s.image_path,
        height=s.height,
        width=s.width,
        fps=1,
        num_frames=s.num_frames,
        seed=s.seed,
        output_path=s.output_path,
        output_file_name=s.output_file_name,
        num_outputs_per_prompt=s.num_outputs_per_prompt,
        save_output=s.save_output,
    )


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
    )
    batch = prepare_request(
        prompt=request.prompt,
        server_args=get_global_server_args(),
        sampling_params=sampling,
    )
    # Run synchronously for images and save to disk
    result = await scheduler_client.forward([batch])
    save_file_path = os.path.join(batch.output_path, batch.output_file_name)
    post_process_sample(
        result.output[0],
        batch.data_type,
        1,
        batch.save_output,
        save_file_path,
    )

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        },
    )

    resp_format = (request.response_format or "b64_json").lower()
    if resp_format == "b64_json":
        with open(save_file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return ImageResponse(
            data=[
                ImageResponseData(
                    b64_json=b64,
                    revised_prompt=request.prompt,
                )
            ]
        )
    else:
        # Return error, not supported
        raise HTTPException(
            status_code=400, detail="response_format=url is not supported"
        )


@router.post("/edits", response_model=ImageResponse)
async def edits(
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form(None),
    size: Optional[str] = Form("1024x1024"),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    user: Optional[str] = Form(None),
):

    request_id = generate_request_id()
    # Resolve images from either `image` or `image[]` (OpenAI SDK sends `image[]` when list is provided)
    images = image or image_array
    if not images or len(images) == 0:
        raise HTTPException(status_code=422, detail="Field 'image' is required")

    # Save first input image; additional images or mask are not yet used by the pipeline
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    first_image = images[0]
    input_path = os.path.join(uploads_dir, f"{request_id}_{first_image.filename}")
    await _save_upload_to_path(first_image, input_path)

    sampling = _build_sampling_params_from_request(
        request_id=request_id,
        prompt=prompt,
        n=n or 1,
        size=size,
        output_format=output_format,
        background=background,
        image_path=input_path,
    )
    batch = _build_req_from_sampling(sampling)

    result = await scheduler_client.forward([batch])
    save_file_path = os.path.join(batch.output_path, batch.output_file_name)
    post_process_sample(
        result.output[0],
        batch.data_type,
        1,
        batch.save_output,
        save_file_path,
    )

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        },
    )

    # Default to b64_json to align with gpt-image-1 behavior in OpenAI examples
    if (response_format or "b64_json").lower() == "b64_json":
        with open(save_file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return ImageResponse(
            data=[ImageResponseData(b64_json=b64, revised_prompt=prompt)]
        )
    else:
        url = f"/v1/images/{request_id}/content"
        return ImageResponse(data=[ImageResponseData(url=url, revised_prompt=prompt)])


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

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
