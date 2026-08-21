# SPDX-License-Identifier: Apache-2.0
"""OpenAI image API adaptation for native SenseNova U1 SRT execution."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import time
import uuid
from collections import OrderedDict

import numpy as np
from fastapi import Request
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImagePromptTokensDetails,
    ImageResponse,
    ImageResponseData,
    ImageUsage,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.models.neo_chat_limits import (
    U1_FLOW_CUSTOM_PARAM,
    U1_IMAGE_CONDITIONING_CUSTOM_PARAM,
    U1_IMAGE_SIZE_DIVISOR,
    parse_u1_int,
    validate_u1_flow_steps,
    validate_u1_image_size,
)

_U1_IMAGE_SYSTEM_PROMPT = (
    "You are an image generation and editing assistant that accurately understands and "
    "executes user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task "
    "requires reasoning, you MUST start with a <think></think> block. Put all reasoning "
    "inside the block using plain text. DO NOT include any image tags. Keep it reasonable "
    "and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no "
    "reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. "
    "Text-to-Image Generation:\n- Generate a high-quality image based on the user's "
    "description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO "
    "NOT introduce elements that contradict or override the user's intent.\n\nB. Image "
    "Editing:\n- Use the provided image(s) as input or reference for modification or "
    "transformation.\n- The result can be an edited image or a new image based on the "
    "reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n- For any visible text in the image, follow the language specified "
    "for the rendered text in the user's description, not the language of the prompt. If "
    "no language is specified, use the user's input language."
)
_U1_PREFIX_CACHE_MAX_ENTRIES = 64
_U1_PREFIX_IDS_CACHE: OrderedDict[str, list[int]] = OrderedDict()


def _parse_u1_image_size(request: ImageGenerationsRequest) -> tuple[int, int]:
    if request.width is not None or request.height is not None:
        if request.width is None or request.height is None:
            raise ValueError("width and height must be provided together")
        return validate_u1_image_size(request.width, request.height)

    size = "1024x1024" if request.size is None else request.size
    try:
        width_text, height_text = size.lower().split("x", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"invalid image size: {size}") from error
    return validate_u1_image_size(width_text, height_text)


def _u1_tensor_bytes_to_png(
    raw_tensor: bytes,
    shape: list[int],
) -> bytes:
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"invalid SenseNova U1 image tensor shape: {shape}")
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    if len(raw_tensor) != expected_bytes:
        raise ValueError(
            "SenseNova U1 image tensor byte length does not match its shape"
        )

    tensor = np.frombuffer(raw_tensor, dtype=np.float16).reshape(shape)
    image = np.clip(tensor[0].astype(np.float32) * 0.5 + 0.5, 0.0, 1.0)
    image = np.rint(image.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _u1_image_prefix(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{_U1_IMAGE_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<img>"
    )


def _u1_next_t_index(
    input_ids: list[int],
    *,
    image_start_token_id: int,
    image_context_token_id: int,
) -> int:
    image_start_shift = [0]
    image_start_shift.extend(
        int(token_id == image_start_token_id) for token_id in input_ids[:-1]
    )
    current = -1
    maximum = -1
    for token_id, shifted_start in zip(
        input_ids,
        image_start_shift,
        strict=True,
    ):
        current += shifted_start + int(token_id != image_context_token_id)
        maximum = max(maximum, current)
    return maximum + 1


def _u1_prefix_cache_key(
    prefix: str,
    image_data: list[bytes] | None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    for image in image_data or []:
        hasher.update(len(image).to_bytes(8, "little"))
        hasher.update(image)
    return hasher.hexdigest()


def _get_cached_u1_prefix(cache_key: str) -> list[int] | None:
    prefix_ids = _U1_PREFIX_IDS_CACHE.pop(cache_key, None)
    if prefix_ids is None:
        return None
    _U1_PREFIX_IDS_CACHE[cache_key] = prefix_ids
    return list(prefix_ids)


def _put_cached_u1_prefix(cache_key: str, prefix_ids: list[int]) -> None:
    _U1_PREFIX_IDS_CACHE.pop(cache_key, None)
    _U1_PREFIX_IDS_CACHE[cache_key] = list(prefix_ids)
    while len(_U1_PREFIX_IDS_CACHE) > _U1_PREFIX_CACHE_MAX_ENTRIES:
        _U1_PREFIX_IDS_CACHE.popitem(last=False)


def _clear_u1_prefix_cache_for_test() -> None:
    _U1_PREFIX_IDS_CACHE.clear()


def _abort_active_requests(tokenizer_manager, active_rids: set[str]) -> None:
    for rid in tuple(active_rids):
        tokenizer_manager.abort_request(rid)


async def _run_one_generate(
    tokenizer_manager,
    request: GenerateReqInput,
    raw_request: Request,
    active_rids: set[str],
) -> dict:
    if await raw_request.is_disconnected():
        raise asyncio.CancelledError("image request client disconnected")

    active_rids.add(request.rid)
    generator = tokenizer_manager.generate_request(request, raw_request)
    try:
        return await generator.__anext__()
    except asyncio.CancelledError:
        tokenizer_manager.abort_request(request.rid)
        raise
    except Exception as error:
        tokenizer_manager.abort_request(request.rid)
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("image request client disconnected") from error
        raise
    finally:
        active_rids.discard(request.rid)
        await generator.aclose()


def _single_seed(seed: int | list[int] | None) -> int:
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError("SenseNova U1 currently supports exactly one seed")
        seed = seed[0]
    parsed_seed = 0 if seed is None else parse_u1_int(seed, name="seed")
    if parsed_seed < 0 or parsed_seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    return parsed_seed


def _validate_u1_image_request(request: ImageGenerationsRequest) -> None:
    n = 1 if request.n is None else parse_u1_int(request.n, name="n")
    if n != 1:
        raise ValueError("SenseNova U1 currently supports n=1")
    if request.guidance_scale is not None:
        guidance_scale = float(request.guidance_scale)
        if not math.isfinite(guidance_scale) or guidance_scale != 1.0:
            raise ValueError("SenseNova U1 currently supports guidance_scale=1")
    if request.num_inference_steps is not None:
        validate_u1_flow_steps(request.num_inference_steps)
    if request.flow_shift is not None:
        flow_shift = float(request.flow_shift)
        if not math.isfinite(flow_shift) or flow_shift <= 0:
            raise ValueError("flow_shift must be a positive finite number")
    if request.output_format not in (None, "png"):
        raise ValueError("SenseNova U1 currently supports output_format=png")


def _u1_flow_response(
    *,
    request: ImageGenerationsRequest,
    request_id: str,
    prefix_tokens: int,
    image_tokens: int,
    flow_result: dict,
    elapsed: float,
    width: int,
    height: int,
) -> ImageResponse:
    meta_info = flow_result["meta_info"]
    tensor_b64 = meta_info["sensenova_u1_flow_image_b64"][0]
    tensor_shape = meta_info["sensenova_u1_flow_image_shape"][0]
    png_bytes = _u1_tensor_bytes_to_png(
        base64.b64decode(tensor_b64),
        tensor_shape,
    )
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    response_format = (
        "b64_json"
        if request.response_format is None
        else str(request.response_format).lower()
    )
    if response_format == "b64_json":
        response_data = ImageResponseData(
            b64_json=png_b64,
            revised_prompt=request.prompt,
            resize=f"{width}x{height}",
        )
    elif response_format == "url":
        response_data = ImageResponseData(
            url=f"data:image/png;base64,{png_b64}",
            revised_prompt=request.prompt,
            resize=f"{width}x{height}",
        )
    else:
        raise ValueError(f"unsupported response_format: {response_format}")

    cached_tokens = int(meta_info.get("cached_tokens", 0))
    flow_compute = meta_info.get("sensenova_u1_flow_compute_seconds")
    if not flow_compute:
        flow_compute = [elapsed]
    return ImageResponse(
        id=request_id,
        data=[response_data],
        inference_time_s=float(flow_compute[0]),
        usage=ImageUsage(
            prompt_tokens=prefix_tokens,
            completion_tokens=image_tokens,
            total_tokens=prefix_tokens + image_tokens,
            prompt_tokens_details=ImagePromptTokensDetails(cached_tokens=cached_tokens),
            image_count=1,
        ),
    )


async def _serve_sensenova_u1_image(
    tokenizer_manager,
    request: ImageGenerationsRequest,
    *,
    image_data: list[bytes] | None,
    raw_request: Request,
) -> ImageResponse:
    if tokenizer_manager.model_config.hf_config.model_type != "neo_chat":
        raise ValueError("the loaded model does not support native SenseNova U1 images")
    _validate_u1_image_request(request)
    if image_data is not None and len(image_data) != 1:
        raise ValueError("SenseNova U1 currently supports one input image")

    width, height = _parse_u1_image_size(request)
    num_steps = validate_u1_flow_steps(
        2 if request.num_inference_steps is None else request.num_inference_steps
    )
    flow_shift = 1.0 if request.flow_shift is None else float(request.flow_shift)
    seed = _single_seed(request.seed)
    tokenizer = tokenizer_manager.tokenizer
    user_prompt = request.prompt if image_data is None else f"<image>\n{request.prompt}"
    prefix = _u1_image_prefix(user_prompt)
    prefix_cache_key = _u1_prefix_cache_key(prefix, image_data)
    prefix_ids = _get_cached_u1_prefix(prefix_cache_key)
    token_width = width // U1_IMAGE_SIZE_DIVISOR
    token_height = height // U1_IMAGE_SIZE_DIVISOR
    image_tokens = token_width * token_height
    image_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    image_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    request_id = f"image-{uuid.uuid4().hex}"
    extra_key = f"sensenova_u1_image:{prefix_cache_key}"
    sampling = {
        "temperature": 0,
        "max_new_tokens": 1,
        "skip_special_tokens": False,
        "no_stop_trim": True,
    }
    if image_data is not None:
        sampling["custom_params"] = {
            U1_IMAGE_CONDITIONING_CUSTOM_PARAM: True,
        }

    active_rids: set[str] = set()
    start = time.perf_counter()
    try:
        if prefix_ids is None and image_data is None:
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            await _run_one_generate(
                tokenizer_manager,
                GenerateReqInput(
                    rid=f"{request_id}-prefix",
                    input_ids=prefix_ids,
                    sampling_params=sampling,
                    extra_key=extra_key,
                ),
                raw_request,
                active_rids,
            )
            _put_cached_u1_prefix(prefix_cache_key, prefix_ids)
        elif prefix_ids is None:
            prefix_result = await _run_one_generate(
                tokenizer_manager,
                GenerateReqInput(
                    rid=f"{request_id}-prefix",
                    text=prefix,
                    image_data=image_data,
                    sampling_params=sampling,
                    extra_key=extra_key,
                    return_prompt_token_ids=True,
                ),
                raw_request,
                active_rids,
            )
            prefix_ids = prefix_result["prompt_token_ids"]
            _put_cached_u1_prefix(prefix_cache_key, prefix_ids)

        flow_sampling = dict(sampling)
        flow_sampling["custom_params"] = {
            **dict(sampling.get("custom_params") or {}),
            U1_FLOW_CUSTOM_PARAM: {
                "width": width,
                "height": height,
                "num_steps": num_steps,
                "seed": seed,
                "image_start": len(prefix_ids),
                "image_tokens": image_tokens,
                "image_t_index": _u1_next_t_index(
                    prefix_ids,
                    image_start_token_id=image_start_token_id,
                    image_context_token_id=image_context_token_id,
                ),
                "token_height": token_height,
                "token_width": token_width,
                "timestep_shift": flow_shift,
                "enable_timestep_shift": True,
                "return_image_tensor": True,
            },
        }
        placeholder_id = tokenizer.eos_token_id
        if placeholder_id is None:
            raise ValueError("SenseNova U1 requires a tokenizer EOS token")
        if image_data is None:
            flow_request = GenerateReqInput(
                rid=f"{request_id}-flow",
                input_ids=[*prefix_ids, *([placeholder_id] * image_tokens)],
                sampling_params=flow_sampling,
                extra_key=extra_key,
            )
        else:
            placeholder_token = tokenizer.convert_ids_to_tokens(placeholder_id)
            flow_request = GenerateReqInput(
                rid=f"{request_id}-flow",
                text=prefix + placeholder_token * image_tokens,
                image_data=image_data,
                sampling_params=flow_sampling,
                extra_key=extra_key,
            )
        flow_result = await _run_one_generate(
            tokenizer_manager,
            flow_request,
            raw_request,
            active_rids,
        )
        elapsed = time.perf_counter() - start
        return _u1_flow_response(
            request=request,
            request_id=request_id,
            prefix_tokens=len(prefix_ids),
            image_tokens=image_tokens,
            flow_result=flow_result,
            elapsed=elapsed,
            width=width,
            height=height,
        )
    except asyncio.CancelledError:
        _abort_active_requests(tokenizer_manager, active_rids)
        raise
    except Exception:
        _abort_active_requests(tokenizer_manager, active_rids)
        raise


async def serve_sensenova_u1_image_generation(
    tokenizer_manager,
    request: ImageGenerationsRequest,
    *,
    raw_request: Request,
) -> ImageResponse:
    return await _serve_sensenova_u1_image(
        tokenizer_manager,
        request,
        image_data=None,
        raw_request=raw_request,
    )


async def serve_sensenova_u1_image_edit(
    tokenizer_manager,
    request: ImageGenerationsRequest,
    *,
    image_data: list[bytes],
    raw_request: Request,
) -> ImageResponse:
    return await _serve_sensenova_u1_image(
        tokenizer_manager,
        request,
        image_data=image_data,
        raw_request=raw_request,
    )


__all__ = [
    "_clear_u1_prefix_cache_for_test",
    "_parse_u1_image_size",
    "_u1_image_prefix",
    "_u1_next_t_index",
    "_u1_tensor_bytes_to_png",
    "serve_sensenova_u1_image_edit",
    "serve_sensenova_u1_image_generation",
]
