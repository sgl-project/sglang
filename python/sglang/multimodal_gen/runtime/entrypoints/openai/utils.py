# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import inspect
import ipaddress
import json
import os
import shutil
import socket
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Union
from urllib.parse import urljoin, urlsplit, urlunsplit

import anyio
import httpx
from fastapi import HTTPException, UploadFile

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
from sglang.multimodal_gen.runtime.utils.common import parse_size
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)
from sglang.multimodal_gen.runtime.utils.trace_wrapper import trace_req

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
_MAX_IMAGE_INPUT_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_REDIRECTS = 5
_REMOTE_IMAGE_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=10.0,
    write=10.0,
    pool=5.0,
)
_REMOTE_IMAGE_CONTENT_TYPES = {
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class ImageURLValidationError(ValueError):
    """Raised when an image URL violates the public input policy."""


class ImageDownloadError(RuntimeError):
    """Raised when an allowed remote image cannot be downloaded."""


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _parse_size_or_raise(size: str) -> tuple[int, int]:
    width, height = parse_size(size)
    if width is None or height is None or width <= 0 or height <= 0:
        raise _bad_request("size must be formatted as positive WIDTHxHEIGHT")
    return width, height


def _validate_positive_int(kwargs: dict[str, Any], name: str) -> None:
    value = kwargs.get(name)
    if value is not None and int(value) <= 0:
        raise _bad_request(f"{name} must be positive")


def flatten_extra_params(payload: Any) -> dict[str, Any]:
    """Promote vLLM-Omni-style extra_params into regular request fields."""
    if not isinstance(payload, dict):
        return {}

    extra_params = payload.pop("extra_params", None)
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params)
        except Exception:
            extra_params = None
    if not isinstance(extra_params, dict):
        if "guardrails" in payload:
            payload.setdefault("use_guardrails", payload["guardrails"])
        return payload

    for key, value in extra_params.items():
        payload.setdefault(key, value)
    if "guardrails" in extra_params:
        payload.setdefault("use_guardrails", extra_params["guardrails"])

    return payload


@contextmanager
def temp_dir_if_disabled(
    configured_path: str | None,
) -> Generator[str, None, None]:
    """Yield *configured_path* when it is set, otherwise create a temporary
    directory that is automatically removed when the context exits."""
    if configured_path is not None:
        os.makedirs(configured_path, exist_ok=True)
        yield configured_path
    else:
        tmp = tempfile.mkdtemp(prefix="sglang_")
        try:
            yield tmp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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
        w, h = _parse_size_or_raise(size)
        # treat None dimensions as unset so parsed size can fill them
        if kwargs.get("width") is None:
            kwargs["width"] = w
        if kwargs.get("height") is None:
            kwargs["height"] = h

    for name in (
        "width",
        "height",
        "num_frames",
        "num_inference_steps",
        "num_outputs_per_prompt",
    ):
        _validate_positive_int(kwargs, name)

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


async def save_image_to_path(
    image: Union[UploadFile, bytes, str],
    target_path: str,
    *,
    prefer_remote_source: bool = False,
) -> str:
    input_path = await _maybe_url_image(
        image, target_path, prefer_remote_source=prefer_remote_source
    )
    if input_path is None:
        input_path = await _save_upload_to_path(image, target_path)
    return input_path


# Helpers
async def _save_upload_to_path(
    upload: Union[UploadFile, bytes], target_path: str
) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if isinstance(upload, bytes):
        content = upload
    elif isinstance(upload, (bytearray, memoryview)):
        content = bytes(upload)
    else:
        read = getattr(upload, "read", None)
        if not callable(read):
            raise TypeError(f"Unsupported image upload type: {type(upload).__name__}")
        content = read()
        if inspect.isawaitable(content):
            content = await content
        if isinstance(content, (bytearray, memoryview)):
            content = bytes(content)
        if not isinstance(content, bytes):
            raise TypeError(
                f"Image upload read() returned {type(content).__name__}, expected bytes"
            )
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


async def _maybe_url_image(
    img_url: str,
    target_path: str,
    *,
    prefer_remote_source: bool = False,
) -> str | None:
    if not isinstance(img_url, str):
        return None

    if img_url.lower().startswith(("http://", "https://")):
        if prefer_remote_source:
            await _validate_remote_image_url(img_url)
            return img_url
        return await _save_url_image_to_path(img_url, target_path)
    elif img_url.startswith("data:image/"):
        if prefer_remote_source:
            return img_url
        return save_base64_image_to_path(
            img_url,
            target_path,
            max_bytes=_MAX_IMAGE_INPUT_BYTES,
        )
    else:
        raise ValueError("Unsupported image url format")


async def _resolve_hostname_addresses(host: str, port: int) -> tuple[str, ...]:
    def resolve() -> tuple[str, ...]:
        records = socket.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        return tuple({record[4][0] for record in records})

    return await anyio.to_thread.run_sync(resolve)


async def _validate_remote_image_url(image_url: str) -> str:
    try:
        parsed = urlsplit(image_url)
        port = parsed.port
    except ValueError as exc:
        raise ImageURLValidationError("Image URL is malformed") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ImageURLValidationError("Image URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ImageURLValidationError("Image URL must not contain credentials")

    host = parsed.hostname
    if not host:
        raise ImageURLValidationError("Image URL must include a hostname")
    if "%" in host:
        raise ImageURLValidationError("Image URL must not contain an IPv6 zone ID")

    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        try:
            addresses = await _resolve_hostname_addresses(
                host,
                port or (443 if scheme == "https" else 80),
            )
        except OSError as exc:
            raise ImageURLValidationError(
                "Image URL hostname could not be resolved"
            ) from exc
        if not addresses:
            raise ImageURLValidationError("Image URL hostname could not be resolved")
    else:
        addresses = (str(literal_address),)

    for address in addresses:
        try:
            resolved_address = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as exc:
            raise ImageURLValidationError(
                "Image URL hostname resolved to an invalid address"
            ) from exc
        if not resolved_address.is_global:
            raise ImageURLValidationError(
                "Image URL must resolve only to public IP addresses"
            )

    return image_url


def _image_url_for_log(image_url: str) -> str:
    parsed = urlsplit(image_url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _remote_image_extension(image_url: str, content_type_header: str) -> str:
    content_type = content_type_header.split(";", 1)[0].strip().lower()
    if content_type in _REMOTE_IMAGE_CONTENT_TYPES:
        return _REMOTE_IMAGE_CONTENT_TYPES[content_type]

    url_ext = os.path.splitext(urlsplit(image_url).path)[1].lower()
    if url_ext == ".jpeg":
        url_ext = ".jpg"
    if content_type == "application/octet-stream" and url_ext in {
        ".bmp",
        ".gif",
        ".jpg",
        ".png",
        ".webp",
    }:
        return url_ext

    raise ImageURLValidationError(
        f"Remote image has unsupported Content-Type: {content_type or 'missing'}"
    )


def _validate_remote_image_content_length(response: httpx.Response) -> None:
    raw_content_length = response.headers.get("content-length")
    if raw_content_length is None:
        return
    try:
        content_length = int(raw_content_length)
    except ValueError as exc:
        raise ImageURLValidationError(
            "Remote image has an invalid Content-Length"
        ) from exc
    if content_length < 0:
        raise ImageURLValidationError("Remote image has an invalid Content-Length")
    if content_length > _MAX_IMAGE_INPUT_BYTES:
        raise ImageURLValidationError(
            f"Remote image exceeds the {_MAX_IMAGE_INPUT_BYTES}-byte limit"
        )


async def _download_remote_image(
    client: httpx.AsyncClient,
    image_url: str,
) -> tuple[bytes, str]:
    current_url = image_url
    redirect_statuses = {301, 302, 303, 307, 308}

    for redirect_count in range(_MAX_IMAGE_REDIRECTS + 1):
        await _validate_remote_image_url(current_url)
        async with client.stream("GET", current_url) as response:
            if response.status_code in redirect_statuses:
                location = response.headers.get("location")
                if not location:
                    raise ImageURLValidationError(
                        "Remote image redirect is missing Location"
                    )
                if redirect_count == _MAX_IMAGE_REDIRECTS:
                    raise ImageURLValidationError(
                        f"Remote image exceeds {_MAX_IMAGE_REDIRECTS} redirects"
                    )
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            extension = _remote_image_extension(
                current_url,
                response.headers.get("content-type", ""),
            )
            _validate_remote_image_content_length(response)

            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > _MAX_IMAGE_INPUT_BYTES:
                    raise ImageURLValidationError(
                        f"Remote image exceeds the {_MAX_IMAGE_INPUT_BYTES}-byte limit"
                    )
            if not content:
                raise ImageURLValidationError("Remote image response is empty")
            return bytes(content), extension

    raise AssertionError("redirect loop terminated without a result")


def _is_retryable_download_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 429 or 500 <= status_code < 600
    return isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    )


async def _save_url_image_to_path(image_url: str, target_path: str) -> str:
    """Download a bounded public image URL to *target_path*."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    max_attempts = 3
    backoff_seconds = 0.2
    last_error: Exception | None = None

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=_REMOTE_IMAGE_TIMEOUT,
            trust_env=False,
        ) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    content, extension = await _download_remote_image(client, image_url)
                    output_path = target_path
                    if not os.path.splitext(output_path)[1]:
                        output_path = f"{output_path}{extension}"
                    with open(output_path, "wb") as file:
                        file.write(content)
                    return output_path
                except ImageURLValidationError:
                    raise
                except Exception as error:
                    last_error = error
                    if attempt == max_attempts or not _is_retryable_download_error(
                        error
                    ):
                        raise
                    wait_s = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Retrying image download (%s/%s) for %s after %.1fs due to %s",
                        attempt,
                        max_attempts,
                        _image_url_for_log(image_url),
                        wait_s,
                        type(error).__name__,
                    )
                    await anyio.sleep(wait_s)
    except ImageURLValidationError:
        raise
    except Exception as error:
        final_error = last_error or error
        if isinstance(final_error, httpx.HTTPStatusError):
            reason = f"HTTP {final_error.response.status_code}"
        else:
            reason = type(final_error).__name__
        raise ImageDownloadError(f"Remote image download failed: {reason}") from error


async def process_generation_batch(
    scheduler_client: AsyncSchedulerClient,
    batch,
) -> tuple[list[str], OutputBatch]:
    total_start_time = time.perf_counter()
    with trace_req(batch.trace_ctx), log_generation_timer(logger, batch.prompt):
        result = await scheduler_client.forward([batch])

        if (
            result.output is None
            and result.output_file_paths is None
            and result.raw_frame_batches is None
            and result.text_outputs is None
        ):
            error_msg = result.error or "Unknown error"
            raise RuntimeError(
                f"Model generation returned no output. Error from scheduler: {error_msg}"
            )

        save_file_path_list = []
        if result.output_file_paths:
            save_file_path_list = result.output_file_paths
        elif result.output is not None:
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
                enable_upscaling=batch.enable_upscaling,
                upscaling_model_path=batch.upscaling_model_path,
                upscaling_scale=batch.upscaling_scale,
            )

    total_time = time.perf_counter() - total_start_time
    if get_global_server_args().batching_max_size > 1:
        completed_outputs = len(result.text_outputs or save_file_path_list)
        log_batch_completion(
            logger,
            completed_outputs,
            total_time,
        )

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

    if result.action_pred is not None:
        t = result.action_pred
        response["action"] = {
            "data": t.tolist(),
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "raw_action_dim": result.action_raw_action_dim,
            "action_mode": result.action_mode,
            "domain_id": result.action_domain_id,
        }

    return response


def adjust_output_quality(output_quality: str, data_type: DataType = None) -> int:
    if output_quality == "default":
        return 50 if data_type == DataType.VIDEO else 75
    return OUTPUT_QUALITY_MAPPER.get(output_quality, None)
