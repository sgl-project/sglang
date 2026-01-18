# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import dataclasses
import ipaddress
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import UploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
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


def sanitize_upload_filename(filename: str, fallback: str) -> str:
    name = os.path.basename(filename or "")
    if not name or name in {".", ".."}:
        name = fallback

    stem, ext = os.path.splitext(name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    safe_ext = re.sub(r"[^A-Za-z0-9.]+", "", ext)
    if not safe_stem:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", fallback).strip("._") or "upload"
    return f"{safe_stem}{safe_ext}"


def ensure_path_within_root(target_path: str, root_dir: str) -> str:
    root_path = Path(root_dir).resolve()
    candidate = Path(target_path).resolve()
    if root_path != candidate and root_path not in candidate.parents:
        raise ValueError("Upload path escapes the uploads root")
    return str(candidate)


def _parse_allowlist(raw_value: str) -> tuple[set[str], list[ipaddress._BaseNetwork]]:
    hosts: set[str] = set()
    networks: list[ipaddress._BaseNetwork] = []
    for entry in (raw_value or "").split(","):
        item = entry.strip()
        if not item:
            continue
        if "/" in item:
            try:
                networks.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                continue
        else:
            hosts.add(item.lower())
    return hosts, networks


def _resolve_host_ips(hostname: str) -> list[ipaddress._BaseAddress]:
    try:
        ip = ipaddress.ip_address(hostname)
        return [ip]
    except ValueError:
        pass

    ips: list[ipaddress._BaseAddress] = []
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            address = sockaddr[0]
            try:
                ips.append(ipaddress.ip_address(address))
            except ValueError:
                continue
    except socket.gaierror:
        return []
    return ips


def _is_ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _is_host_allowlisted(
    hostname: str,
    ips: list[ipaddress._BaseAddress],
    allow_hosts: set[str],
    allow_nets: list[ipaddress._BaseNetwork],
) -> bool:
    if hostname in allow_hosts:
        return True
    for ip in ips:
        if any(ip in net for net in allow_nets):
            return True
    return False


def _get_openai_media_url_policy() -> dict[str, Any]:
    enabled = get_bool_env_var("SGLANG_OPENAI_MEDIA_URL_FETCH_ENABLED", "true")
    allowed_schemes = os.getenv("SGLANG_OPENAI_MEDIA_URL_ALLOWED_SCHEMES", "https,http")
    allowlist_raw = os.getenv("SGLANG_OPENAI_MEDIA_URL_ALLOWLIST", "")
    default_max_bytes = 50 * 1024 * 1024
    default_max_redirects = 5
    default_timeout = 10.0
    try:
        max_bytes = int(
            os.getenv("SGLANG_OPENAI_MEDIA_URL_MAX_BYTES", str(default_max_bytes))
        )
    except (TypeError, ValueError):
        max_bytes = default_max_bytes
    try:
        max_redirects = int(
            os.getenv(
                "SGLANG_OPENAI_MEDIA_URL_MAX_REDIRECTS", str(default_max_redirects)
            )
        )
    except (TypeError, ValueError):
        max_redirects = default_max_redirects
    try:
        timeout = float(
            os.getenv("SGLANG_OPENAI_MEDIA_URL_TIMEOUT", str(default_timeout))
        )
    except (TypeError, ValueError):
        timeout = default_timeout

    schemes = {
        scheme.strip().lower()
        for scheme in allowed_schemes.split(",")
        if scheme.strip()
    }
    if not schemes:
        schemes = {"https"}
    allow_hosts, allow_nets = _parse_allowlist(allowlist_raw)
    return {
        "enabled": enabled,
        "schemes": schemes,
        "allow_hosts": allow_hosts,
        "allow_nets": allow_nets,
        "max_bytes": max_bytes,
        "max_redirects": max_redirects,
        "timeout": timeout,
    }


def _validate_media_url_with_policy(url: str, policy: dict[str, Any]) -> None:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in policy["schemes"]:
        raise ValueError(f"URL scheme '{scheme}' is not allowed")
    if not hostname:
        raise ValueError("URL host is missing")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        if hostname not in policy["allow_hosts"]:
            raise ValueError("Localhost URLs are not allowed")

    ips = _resolve_host_ips(hostname)
    if not ips:
        raise ValueError("Failed to resolve URL host")
    if _is_host_allowlisted(hostname, ips, policy["allow_hosts"], policy["allow_nets"]):
        return
    for ip in ips:
        if _is_ip_blocked(ip):
            raise ValueError("URL resolves to a private or reserved IP")


def validate_openai_media_url(url: str) -> dict[str, Any]:
    policy = _get_openai_media_url_policy()
    if not policy["enabled"]:
        raise ValueError("Remote media URL fetching is disabled by configuration")
    _validate_media_url_with_policy(url, policy)
    return policy


async def save_image_to_path(
    image: Union[UploadFile, str],
    target_path: str,
    uploads_root: str | None = None,
) -> str:
    if uploads_root:
        target_path = ensure_path_within_root(target_path, uploads_root)
    input_path = await _maybe_url_image(image, target_path, uploads_root=uploads_root)
    if input_path is None:
        input_path = await _save_upload_to_path(
            image, target_path, uploads_root=uploads_root
        )
    return input_path


# Helpers
async def _save_upload_to_path(
    upload: UploadFile,
    target_path: str,
    uploads_root: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


async def _maybe_url_image(
    img_url: str,
    target_path: str,
    uploads_root: str | None = None,
) -> str | None:
    if not isinstance(img_url, str):
        return None

    if img_url.lower().startswith(("http://", "https://")):
        policy = validate_openai_media_url(img_url)
        input_path = await _save_url_image_to_path(
            img_url, target_path, policy, uploads_root=uploads_root
        )
        return input_path
    elif img_url.startswith("data:image"):
        # encode image base64 url
        input_path = await _save_base64_image_to_path(
            img_url, target_path, uploads_root=uploads_root
        )
        return input_path
    else:
        raise ValueError("Unsupported image url format")


async def _save_url_image_to_path(
    image_url: str,
    target_path: str,
    policy: dict[str, Any] | None = None,
    uploads_root: str | None = None,
) -> str:
    """Download image from URL and save to target path."""
    try:
        policy = policy or validate_openai_media_url(image_url)
        current_url = image_url

        async with httpx.AsyncClient(follow_redirects=False) as client:
            for _ in range(policy["max_redirects"] + 1):
                _validate_media_url_with_policy(current_url, policy)
                async with client.stream(
                    "GET", current_url, timeout=policy["timeout"]
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError(
                                "Redirect response missing location header"
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()

                    # Determine file extension from content type or URL after downloading
                    if not os.path.splitext(target_path)[1]:
                        content_type = response.headers.get("content-type", "").lower()

                        url_path = current_url.split("?")[0]
                        _, url_ext = os.path.splitext(url_path)
                        url_ext = url_ext.lower()

                        if url_ext in {
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".webp",
                            ".gif",
                            ".bmp",
                        }:
                            ext = ".jpg" if url_ext == ".jpeg" else url_ext
                        elif content_type.startswith("image/"):
                            if "jpeg" in content_type or "jpg" in content_type:
                                ext = ".jpg"
                            elif "png" in content_type:
                                ext = ".png"
                            elif "webp" in content_type:
                                ext = ".webp"
                            else:
                                ext = ".jpg"
                        elif content_type == "application/octet-stream":
                            ext = ".jpg"
                        else:
                            raise ValueError(
                                f"URL does not point to an image. Content-Type: {content_type}"
                            )
                        target_path = f"{target_path}{ext}"

                    if uploads_root:
                        target_path = ensure_path_within_root(target_path, uploads_root)

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > policy["max_bytes"]:
                        raise ValueError("Remote content exceeds max size limit")

                    total = 0
                    with open(target_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > policy["max_bytes"]:
                                raise ValueError(
                                    "Remote content exceeds max size limit"
                                )
                            f.write(chunk)
                    return target_path

            raise ValueError("Too many redirects when fetching media URL")
    except Exception as e:
        raise Exception(f"Failed to download image from URL: {str(e)}")


async def _save_base64_image_to_path(
    base64_data: str,
    target_path: str,
    uploads_root: str | None = None,
) -> str:
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
    if uploads_root:
        target_path = ensure_path_within_root(target_path, uploads_root)
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
        save_file_path_list = []
        if batch.data_type == DataType.VIDEO:
            for idx, output in enumerate(result.output):
                save_file_path = str(
                    os.path.join(batch.output_path, batch.output_file_name)
                )
                post_process_sample(
                    result.output[idx],
                    batch.data_type,
                    batch.fps,
                    batch.save_output,
                    save_file_path,
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
