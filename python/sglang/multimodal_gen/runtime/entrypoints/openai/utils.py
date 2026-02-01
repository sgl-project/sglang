# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import dataclasses
import os
import re
import time
from typing import Any, List, Optional, Union

import httpx
import torch
from fastapi import UploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)

logger = init_logger(__name__)


@dataclasses.dataclass
class SetLoraReq:
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0  # LoRA strength for merge, default 1.0


@dataclasses.dataclass
class MergeLoraWeightsReq:
    target: str = "all"  # "all", "transformer", "transformer_2", "critic"
    strength: float = 1.0  # LoRA strength for merge, default 1.0


@dataclasses.dataclass
class UnmergeLoraWeightsReq:
    target: str = "all"  # "all", "transformer", "transformer_2", "critic"


@dataclasses.dataclass
class ListLorasReq:
    # Empty payload; used only as a type marker for listing LoRA status
    pass


def format_lora_message(
    lora_nickname: Union[str, List[str]],
    target: Union[str, List[str]],
    strength: Union[float, List[float]],
) -> tuple[str, str, str]:
    """Format success message for single or multiple LoRAs"""
    if isinstance(lora_nickname, list):
        nickname_str = ", ".join(lora_nickname)
        target_str = ", ".join(target) if isinstance(target, list) else target
        strength_str = (
            ", ".join(f"{s:.2f}" for s in strength)
            if isinstance(strength, list)
            else f"{strength:.2f}"
        )
    else:
        nickname_str = lora_nickname
        target_str = target if isinstance(target, str) else ", ".join(target)
        strength_str = (
            f"{strength:.2f}"
            if isinstance(strength, (int, float))
            else ", ".join(f"{s:.2f}" for s in strength)
        )
    return nickname_str, target_str, strength_str


def _parse_size(size: str) -> tuple[int, int] | tuple[None, None]:
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        return None, None


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
            error_msg = result.error or "Unknown error"
            raise RuntimeError(
                f"Model generation returned no output. Error from scheduler: {error_msg}"
            )
        save_file_path_list = []
        audio_sample_rate = result.audio_sample_rate
        if batch.data_type == DataType.VIDEO:
            for idx, output in enumerate(result.output):
                save_file_path = str(
                    os.path.join(batch.output_path, batch.output_file_name)
                )
                sample = result.output[idx]
                audio = result.audio
                if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
                    audio = audio[idx] if audio.shape[0] > idx else None
                if audio is not None and not (
                    isinstance(sample, (tuple, list)) and len(sample) == 2
                ):
                    sample = (sample, audio)
                post_process_sample(
                    sample,
                    batch.data_type,
                    batch.fps,
                    batch.save_output,
                    save_file_path,
                    audio_sample_rate=audio_sample_rate,
                )
                save_file_path_list.append(save_file_path)
        else:
            for idx, output in enumerate(result.output):
                save_file_path = str(
                    os.path.join(
                        batch.output_path, f"sample_{idx}_" + batch.output_file_name
                    )
                )
                post_process_sample(
                    output,
                    batch.data_type,
                    batch.fps,
                    batch.save_output,
                    save_file_path,
                    audio_sample_rate=audio_sample_rate,
                )
                save_file_path_list.append(save_file_path)

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

    if result.timings and result.timings.total_duration_s > 0:
        response["inference_time_s"] = result.timings.total_duration_s

    response["id"] = request_id

    return response
