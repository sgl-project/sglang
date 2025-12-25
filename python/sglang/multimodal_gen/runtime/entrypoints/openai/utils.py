# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import os

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from fastapi import UploadFile

from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)

logger = init_logger(__name__)


def post_process_sample(
    sample: torch.Tensor,
    data_type: DataType,
    fps: int,
    save_output: bool = True,
    save_file_path: str = None,
):
    """
    Process sample output and save video if necessary
    """
    # Process outputs
    if sample.dim() == 3:
        # for images, dim t is missing
        sample = sample.unsqueeze(1)
    videos = rearrange(sample, "c t h w -> t c h w")
    frames = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    # Save outputs if requested
    if save_output:
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if data_type == DataType.VIDEO:
                imageio.mimsave(
                    save_file_path,
                    frames,
                    fps=fps,
                    format=data_type.get_default_extension(),
                )
            else:
                imageio.imwrite(save_file_path, frames[0])
            logger.info(f"Saved output to {save_file_path}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames


def _parse_size(size: str) -> tuple[int, int]:
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        # Fallback to default portrait 720x1280
        return 720, 1280


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

    # split `data:[<media-type>][;base64],<data>` to media-type base64 data
    pattern = r"data:(.*?)(;base64)?,(.*)"
    match = re.match(pattern, base64_data)
    if not match:
        raise ValueError(
            f"Failed to decoding base64 image, please make sure the url format `data:[<media-type>][;base64],<data>` "
        )
    media_type = match.group(1)
    is_base64 = match.group(2)
    if not is_base64:
        raise ValueError(
            f"Failed to decoding base64 image, please make sure the url format `data:[<media-type>][;base64],<data>` "
        )
    data = match.group(3)
    if not data:
        raise ValueError(
            f"Failed to decoding base64 image, please make sure the url format `data:[<media-type>][;base64],<data>` "
        )
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
) -> tuple[str, OutputBatch]:
    total_start_time = time.perf_counter()
    with log_generation_timer(logger, batch.prompt):
        result = await scheduler_client.forward([batch])

        if result.output is None:
            error_msg = getattr(result, "error", "Unknown error")
            raise RuntimeError(
                f"Model generation returned no output. Error from scheduler: {error_msg}"
            )

        save_file_path = str(os.path.join(batch.output_path, batch.output_file_name))
        post_process_sample(
            result.output[0],
            batch.data_type,
            batch.fps,
            batch.save_output,
            save_file_path,
        )

    total_time = time.perf_counter() - total_start_time
    log_batch_completion(logger, 1, total_time)

    if result.peak_memory_mb and result.peak_memory_mb > 0:
        logger.info(f"Peak memory usage: {result.peak_memory_mb:.2f} MB")

    return save_file_path, result


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


def add_common_data_to_response(response: dict, request_id: str, result: OutputBatch) -> dict:
    if result.peak_memory_mb and result.peak_memory_mb > 0:
        response["peak_memory_mb"] = result.peak_memory_mb

    response.update(dict(rid=request_id))

    return response
