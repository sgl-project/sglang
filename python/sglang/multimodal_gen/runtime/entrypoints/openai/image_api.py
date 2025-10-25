import asyncio
import base64
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sgl_diffusion.api.configs.sample.base import (
    SamplingParams,
    generate_request_id,
)
from sgl_diffusion.runtime.entrypoints.openai.utils import _parse_size
from sgl_diffusion.runtime.entrypoints.utils import prepare_request
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import Req
from sgl_diffusion.runtime.scheduler_client import scheduler_client
from sgl_diffusion.runtime.server_args import get_global_server_args
from sgl_diffusion.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1/images", tags=["images"])
logger = init_logger(__name__)

# In-memory store for produced images (non-persistent)
IMAGE_ITEMS: Dict[str, Dict[str, Any]] = {}
IMAGE_LOCK = asyncio.Lock()


# TODO: move this to `types.py`
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


async def _save_upload_to_path(upload: UploadFile, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


from sgl_diffusion.runtime.entrypoints.openai.utils import post_process_sample


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

    async with IMAGE_LOCK:
        IMAGE_ITEMS[request_id] = {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        }

    # TODO: verify this first.
    if (request.response_format or "b64_json").lower() == "b64_json":
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
    image: List[UploadFile] = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("url"),
    size: Optional[str] = Form("1024x1024"),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    user: Optional[str] = Form(None),
):

    request_id = generate_request_id()
    # Save first input image; additional images or mask are not yet used by the pipeline
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    first_image = image[0]
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

    async with IMAGE_LOCK:
        IMAGE_ITEMS[request_id] = {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        }

    if (response_format or "url").lower() == "b64_json":
        with open(save_file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return ImageResponse(
            data=[ImageResponseData(b64_json=b64, revised_prompt=prompt)]
        )
    else:
        url = f"/v1/images/{request_id}/content"
        return ImageResponse(data=[ImageResponseData(url=url, revised_prompt=prompt)])


@router.post("/variations", response_model=ImageResponse)
async def variations(
    image: UploadFile = File(...),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("url"),
    size: Optional[str] = Form("1024x1024"),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    user: Optional[str] = Form(None),
):

    request_id = generate_request_id()
    uploads_dir = os.path.join("outputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    input_path = os.path.join(uploads_dir, f"{request_id}_{image.filename}")
    await _save_upload_to_path(image, input_path)

    sampling = _build_sampling_params_from_request(
        request_id=request_id,
        prompt="",  # variations do not require a prompt
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

    async with IMAGE_LOCK:
        IMAGE_ITEMS[request_id] = {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": save_file_path,
        }

    if (response_format or "url").lower() == "b64_json":
        with open(save_file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return ImageResponse(data=[ImageResponseData(b64_json=b64)])
    else:
        url = f"/v1/images/{request_id}/content"
        return ImageResponse(data=[ImageResponseData(url=url)])


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    async with IMAGE_LOCK:
        item = IMAGE_ITEMS.get(image_id)
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
