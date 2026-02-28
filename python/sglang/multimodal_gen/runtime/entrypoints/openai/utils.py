# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import os
import re
import time
from typing import Any, List, Optional, Union

import httpx
from fastapi import UploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
    save_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)

# re-export LoRA protocol types for backward compatibility
__all__ = [
    "SetLoraReq",
    "MergeLoraWeightsReq",
    "UnmergeLoraWeightsReq",
    "ListLorasReq",
    "ShutdownReq",
    "format_lora_message",
]

logger = init_logger(__name__)

OUTPUT_QUALITY_MAPPER = {"maximum": 100, "high": 90, "medium": 55, "low": 35}
DEFAULT_FPS = 24
DEFAULT_VIDEO_SECONDS = 4


def _parse_size(size: str) -> tuple[int, int] | tuple[None, None]:
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        return None, None


def choose_output_image_ext(
    output_format: Optional[str], background: Optional[str]
) -> str:
    fmt = (output_format or "").lower()
    if fmt in {"png", "webp", "jpeg", "jpg"}:
        return "jpg" if fmt == "jpeg" else fmt
    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpg"


def build_sampling_params(request_id: str, **kwargs) -> SamplingParams:
    """Build SamplingParams from request parameters.

    Handles size parsing, output_quality resolution, and None filtering before
    delegating to SamplingParams.from_user_sampling_params_args. Callers pass
    only the parameters they have; None values are stripped automatically so
    that SamplingParams defaults apply.
    """
    server_args = get_global_server_args()

    # pop HTTP-layer params that aren't SamplingParams fields
    output_quality = kwargs.pop("output_quality", None)

    has_explicit_compression = kwargs.get("output_compression") is not None

    # parse "WxH" size string if provided
    size = kwargs.pop("size", None)
    if size:
        w, h = _parse_size(size)
        if w is not None:
            kwargs.setdefault("width", w)
            kwargs.setdefault("height", h)

    # filter out None values to let SamplingParams defaults apply
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.setdefault("save_output", True)

    sampling_params = SamplingParams.from_user_sampling_params_args(
        model_path=server_args.model_path,
        server_args=server_args,
        request_id=request_id,
        **kwargs,
    )

    # resolve output_quality → output_compression with the correct data_type.
    # SamplingParams.__post_init__ may have resolved with the wrong data_type
    # (default VIDEO) before _adjust() set the correct one.
    if not has_explicit_compression and output_quality is not None:
        resolved = adjust_output_quality(output_quality, sampling_params.data_type)
        if resolved is not None:
            sampling_params.output_compression = resolved

    return sampling_params


async def save_image_to_path(image: Union[UploadFile, str], target_path: str) -> str:
    input_path = await _maybe_url_image(image, target_path)
    if input_path is None:
        input_path = await _save_upload_to_path(image, target_path)
    return input_path


# Helpers
async def _save_upload_to_path(upload: UploadFile, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


async def _maybe_url_image(img_url: str, target_path: str) -> str | None:
    if not isinstance(img_url, str):
        return None

    if img_url.lower().startswith(("http://", "https://")):
        # Download image from URL
        input_path = await _save_url_image_to_path(img_url, target_path)
        return input_path
    elif img_url.startswith("data:image"):
        # encode image base64 url
        input_path = await _save_base64_image_to_path(img_url, target_path)
        return input_path
    else:
        raise ValueError("Unsupported image url format")


async def _save_url_image_to_path(image_url: str, target_path: str) -> str:
    """Download image from URL and save to target path."""

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(image_url, timeout=10.0)
            response.raise_for_status()

            # Determine file extension from content type or URL after downloading
            if not os.path.splitext(target_path)[1]:
                content_type = response.headers.get("content-type", "").lower()

                url_path = image_url.split("?")[0]
                _, url_ext = os.path.splitext(url_path)
                url_ext = url_ext.lower()

                if url_ext in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                    ext = ".jpg" if url_ext == ".jpeg" else url_ext
                elif content_type.startswith("image/"):
                    if "jpeg" in content_type or "jpg" in content_type:
                        ext = ".jpg"
                    elif "png" in content_type:
                        ext = ".png"
                    elif "webp" in content_type:
                        ext = ".webp"
                    else:
                        ext = ".jpg"  # Default to jpg
                elif content_type == "application/octet-stream":
                    # for octet-stream, if we couldn't get it from URL, default to jpg
                    ext = ".jpg"
                else:
                    raise ValueError(
                        f"URL does not point to an image. Content-Type: {content_type}"
                    )
                target_path = f"{target_path}{ext}"

            with open(target_path, "wb") as f:
                f.write(response.content)

            return target_path
    except Exception as e:
        raise Exception(f"Failed to download image from URL: {str(e)}")


async def _save_base64_image_to_path(base64_data: str, target_path: str) -> str:
    """Decode base64 image data and save to target path."""

    _B64_FMT_HINT = (
        "Failed to decode base64 image. "
        "Expected format: `data:[<media-type>];base64,<data>`"
    )

    # split `data:[<media-type>][;base64],<data>` to media-type base64 data
    pattern = r"data:(.*?)(;base64)?,(.*)"
    match = re.match(pattern, base64_data)
    if not match:
        raise ValueError(_B64_FMT_HINT)
    media_type = match.group(1)
    is_base64 = match.group(2)
    if not is_base64:
        raise ValueError(f"{_B64_FMT_HINT} (missing ;base64 marker)")
    data = match.group(3)
    if not data:
        raise ValueError(f"{_B64_FMT_HINT} (empty data payload)")
    # get ext from url
    if media_type.startswith("image/"):
        ext = media_type.split("/")[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
    else:
        ext = "jpg"
    target_path = f"{target_path}.{ext}"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        image_data = base64.b64decode(data)
        with open(target_path, "wb") as f:
            f.write(image_data)

        return target_path
    except Exception as e:
        raise Exception(f"Failed to decode base64 image: {str(e)}")


async def process_generation_batch(
    scheduler_client: AsyncSchedulerClient,
    batch,
) -> tuple[list[str], OutputBatch]:
    total_start_time = time.perf_counter()
    with log_generation_timer(logger, batch.prompt):
        result = await scheduler_client.forward([batch])

        if result.output is None and result.output_file_paths is None:
            error_msg = result.error or "Unknown error"
            raise RuntimeError(
                f"Model generation returned no output. Error from scheduler: {error_msg}"
            )

        if result.output_file_paths:
            save_file_path_list = result.output_file_paths
        else:
            num_outputs = len(result.output)
            save_file_path_list = save_outputs(
                result.output,
                batch.data_type,
                batch.fps,
                batch.save_output,
                lambda idx: str(batch.output_file_path(num_outputs, idx)),
                audio=result.audio,
                audio_sample_rate=result.audio_sample_rate,
                output_compression=batch.output_compression,
                enable_frame_interpolation=batch.enable_frame_interpolation,
                frame_interpolation_exp=batch.frame_interpolation_exp,
                frame_interpolation_scale=batch.frame_interpolation_scale,
                frame_interpolation_model_path=batch.frame_interpolation_model_path,
            )

    total_time = time.perf_counter() - total_start_time
    log_batch_completion(logger, 1, total_time)

    if result.peak_memory_mb and result.peak_memory_mb > 0:
        logger.info(f"Peak memory usage: {result.peak_memory_mb:.2f} MB")

    return save_file_path_list, result


def merge_image_input_list(*inputs: Union[List, Any, None]) -> List:
    """
    Merge multiple image input sources into a single list.

    This function handles both single items and lists of items, merging them
    into a single flattened list. Useful for processing images, URLs, or other
    multimedia inputs that can come as either single items or lists.

    Args:
        *inputs: Variable number of inputs, each can be None, single item, or list

    Returns:
        List: Flattened list of all non-None inputs

    Example:
        >>> merge_image_input_list(["img1", "img2"], "img3", None)
        ["img1", "img2", "img3"]
    """
    result = []
    for input_item in inputs:
        if input_item is not None:
            if isinstance(input_item, list):
                result.extend(input_item)
            else:
                result.append(input_item)
    return result


def add_common_data_to_response(
    response: dict, request_id: str, result: OutputBatch
) -> dict:
    if result.peak_memory_mb and result.peak_memory_mb > 0:
        response["peak_memory_mb"] = result.peak_memory_mb

    if result.metrics and result.metrics.total_duration_s > 0:
        response["inference_time_s"] = result.metrics.total_duration_s

    response["id"] = request_id

    return response


def adjust_output_quality(output_quality: str, data_type: DataType = None) -> int:
    if output_quality == "default":
        return 50 if data_type == DataType.VIDEO else 75
    return OUTPUT_QUALITY_MAPPER.get(output_quality, None)
