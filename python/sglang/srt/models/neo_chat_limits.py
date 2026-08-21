# SPDX-License-Identifier: Apache-2.0
"""Serving limits shared by SenseNova U1 image-generation paths."""

from __future__ import annotations

import math
from typing import Any

U1_FLOW_CUSTOM_PARAM = "sensenova_u1_flow"
U1_FLOW_BATCH_ISOLATION_PARAM = "__sglang_batch_isolation_key"
U1_FLOW_RADIX_PREFIX_LIMIT_PARAM = "__sglang_radix_cache_prefix_limit"
U1_FLOW_PREFILL_GRAPH_VARIANT_PARAM = "__sglang_prefill_cuda_graph_variant"
U1_INTERLEAVE_CUSTOM_PARAM = "sensenova_u1_interleave"
U1_EXACT_TEXT_CUSTOM_PARAM = "sensenova_u1_exact_text"
U1_IMAGE_CONDITIONING_CUSTOM_PARAM = "sensenova_u1_image_conditioning"
U1_IMAGE_CONDITIONING_MIN_PIXELS = 512 * 512
U1_IMAGE_CONDITIONING_MAX_PIXELS = 2048 * 2048
U1_IMAGE_SIZE_DIVISOR = 32
U1_MAX_IMAGE_DIMENSION = 2048
U1_MAX_IMAGE_PIXELS = 1024 * 1024
U1_MAX_FLOW_STEPS = 64
U1_MAX_INTERLEAVE_IMAGES = 8


def derive_u1_turn_seeds(seed: int, max_images: int) -> tuple[int, ...]:
    return tuple((seed + image_index) % (2**63) for image_index in range(max_images))


def parse_u1_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def validate_u1_image_size(width: Any, height: Any) -> tuple[int, int]:
    parsed_width = parse_u1_int(width, name="width")
    parsed_height = parse_u1_int(height, name="height")
    for name, value in (
        ("width", parsed_width),
        ("height", parsed_height),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > U1_MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"{name} exceeds the maximum {U1_MAX_IMAGE_DIMENSION}: {value}"
            )
        if value % U1_IMAGE_SIZE_DIVISOR:
            raise ValueError(
                f"{name} must be divisible by {U1_IMAGE_SIZE_DIVISOR}: {value}"
            )
    pixels = parsed_width * parsed_height
    if pixels > U1_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image pixel count exceeds the maximum {U1_MAX_IMAGE_PIXELS}: {pixels}"
        )
    return parsed_width, parsed_height


def validate_u1_flow_steps(value: Any) -> int:
    steps = parse_u1_int(value, name="num_steps")
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    if steps > U1_MAX_FLOW_STEPS:
        raise ValueError(f"num_steps exceeds the maximum {U1_MAX_FLOW_STEPS}: {steps}")
    return steps


def normalize_u1_flow_request(
    spec: Any,
    *,
    input_token_count: int,
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("sensenova_u1_flow must be an object")

    width, height = validate_u1_image_size(
        spec.get("width"),
        spec.get("height"),
    )
    num_steps = validate_u1_flow_steps(spec.get("num_steps", 2))
    token_width = width // U1_IMAGE_SIZE_DIVISOR
    token_height = height // U1_IMAGE_SIZE_DIVISOR
    image_tokens = token_width * token_height

    image_start = parse_u1_int(spec.get("image_start"), name="image_start")
    image_t_index = parse_u1_int(
        spec.get("image_t_index"),
        name="image_t_index",
    )
    requested_image_tokens = parse_u1_int(
        spec.get("image_tokens"),
        name="image_tokens",
    )
    requested_token_height = parse_u1_int(
        spec.get("token_height"),
        name="token_height",
    )
    requested_token_width = parse_u1_int(
        spec.get("token_width"),
        name="token_width",
    )
    if image_start < 0 or image_t_index < 0:
        raise ValueError("image_start and image_t_index must be non-negative")
    if requested_image_tokens != image_tokens:
        raise ValueError("image_tokens does not match the requested image size")
    if (requested_token_height, requested_token_width) != (
        token_height,
        token_width,
    ):
        raise ValueError("flow token grid does not match the requested image size")
    if image_start + image_tokens != input_token_count:
        raise ValueError("the flow image block must be the final input token span")

    seed = parse_u1_int(spec.get("seed", 0), name="seed")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    timestep_shift = float(spec.get("timestep_shift", 1.0))
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep_shift must be a positive finite number")
    enable_timestep_shift = spec.get("enable_timestep_shift", True)
    return_image_tensor = spec.get("return_image_tensor", False)
    if not isinstance(enable_timestep_shift, bool):
        raise TypeError("enable_timestep_shift must be a boolean")
    if not isinstance(return_image_tensor, bool):
        raise TypeError("return_image_tensor must be a boolean")

    return {
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "seed": seed,
        "image_start": image_start,
        "image_tokens": image_tokens,
        "image_t_index": image_t_index,
        "token_height": token_height,
        "token_width": token_width,
        "timestep_shift": timestep_shift,
        "enable_timestep_shift": enable_timestep_shift,
        "return_image_tensor": return_image_tensor,
    }


def normalize_u1_interleave_request(
    spec: Any,
    *,
    input_token_count: int,
    max_new_tokens: int,
    context_len: int,
    img_start_token_id: int,
    img_context_token_id: int,
    img_end_token_id: int,
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("sensenova_u1_interleave must be an object")

    width, height = validate_u1_image_size(
        spec.get("width", 256),
        spec.get("height", 256),
    )
    num_steps = validate_u1_flow_steps(spec.get("num_steps", 2))
    max_images = parse_u1_int(spec.get("max_images", 1), name="max_images")
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if max_images > U1_MAX_INTERLEAVE_IMAGES:
        raise ValueError(
            f"max_images exceeds the maximum {U1_MAX_INTERLEAVE_IMAGES}: {max_images}"
        )

    seed = parse_u1_int(spec.get("seed", 0), name="seed")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    timestep_shift = float(spec.get("timestep_shift", 1.0))
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep_shift must be a positive finite number")
    enable_timestep_shift = spec.get("enable_timestep_shift", True)
    return_images = spec.get("return_images", True)
    if not isinstance(enable_timestep_shift, bool):
        raise TypeError("enable_timestep_shift must be a boolean")
    if not isinstance(return_images, bool):
        raise TypeError("return_images must be a boolean")
    turn_seeds = derive_u1_turn_seeds(seed, max_images)

    token_width = width // U1_IMAGE_SIZE_DIVISOR
    token_height = height // U1_IMAGE_SIZE_DIVISOR
    image_tokens = token_width * token_height
    image_span_tokens = image_tokens + 1  # Context tokens plus </img>.
    reserved_tokens = (
        input_token_count + max_new_tokens + max_images * image_span_tokens
    )
    if reserved_tokens > context_len:
        raise ValueError(
            "SenseNova U1 interleave request exceeds the context window after "
            f"reserving image spans: {reserved_tokens} > {context_len}"
        )

    return {
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "max_images": max_images,
        "seed": seed,
        "turn_seeds": list(turn_seeds),
        "timestep_shift": timestep_shift,
        "enable_timestep_shift": enable_timestep_shift,
        "return_images": return_images,
        "image_tokens": image_tokens,
        "image_span_tokens": image_span_tokens,
        "token_height": token_height,
        "token_width": token_width,
        "img_start_token_id": int(img_start_token_id),
        "img_context_token_id": int(img_context_token_id),
        "img_end_token_id": int(img_end_token_id),
    }


__all__ = [
    "U1_EXACT_TEXT_CUSTOM_PARAM",
    "U1_FLOW_BATCH_ISOLATION_PARAM",
    "U1_FLOW_CUSTOM_PARAM",
    "U1_FLOW_PREFILL_GRAPH_VARIANT_PARAM",
    "U1_FLOW_RADIX_PREFIX_LIMIT_PARAM",
    "U1_IMAGE_CONDITIONING_CUSTOM_PARAM",
    "U1_IMAGE_CONDITIONING_MAX_PIXELS",
    "U1_IMAGE_CONDITIONING_MIN_PIXELS",
    "U1_IMAGE_SIZE_DIVISOR",
    "U1_INTERLEAVE_CUSTOM_PARAM",
    "U1_MAX_FLOW_STEPS",
    "U1_MAX_IMAGE_DIMENSION",
    "U1_MAX_IMAGE_PIXELS",
    "U1_MAX_INTERLEAVE_IMAGES",
    "derive_u1_turn_seeds",
    "normalize_u1_flow_request",
    "normalize_u1_interleave_request",
    "parse_u1_int",
    "validate_u1_flow_steps",
    "validate_u1_image_size",
]
