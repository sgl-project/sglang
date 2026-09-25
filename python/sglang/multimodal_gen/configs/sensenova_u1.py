# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

import json
import os

from PIL import Image

SENSENOVA_U1_REQUEST_EXTRA_KEY = "sensenova_u1"

SENSENOVA_U1_MODEL_IDS = {
    "sensenova/sensenova-u1.5-8b-mot",
}
SENSENOVA_U1_ADAPTER_ONLY_MODEL_IDS = {
    "sensenova/sensenova-u1.5-8b-mot-loras",
}

SENSENOVA_U1_CFG_NORM_CHOICES = (
    "none",
    "global",
    "channel",
    "cfg_zero_star",
)
RESOLUTION_ALIGNMENT = 32
DEFAULT_EDIT_LONG_SIDE = 2048
MIN_EDIT_SHORT_SIDE = 512
MIN_INPUT_MAX_PIXELS = 512 * 512
SENSENOVA_U1_SIZE_FIELDS = frozenset({"size", "width", "height"})

DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT = True
DEFAULT_CFG_INTERVAL = (0.0, 1.0)
DEFAULT_T_EPS = 0.02
DEFAULT_THINK_MODE = False


def _flatten_rgba_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    return background


def _round_half_up_by_factor(value: float, factor: int) -> int:
    return int(value / factor + 0.5) * factor


def has_sensenova_u1_explicit_size(explicit_fields) -> bool:
    return bool(set(explicit_fields or ()).intersection(SENSENOVA_U1_SIZE_FIELDS))


def resolve_sensenova_u1_edit_auto_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {width}x{height}.")

    if width >= height:
        resized_width = DEFAULT_EDIT_LONG_SIDE
        resized_height = max(
            MIN_EDIT_SHORT_SIDE,
            _round_half_up_by_factor(
                height / width * DEFAULT_EDIT_LONG_SIDE,
                RESOLUTION_ALIGNMENT,
            ),
        )
    else:
        resized_width = max(
            MIN_EDIT_SHORT_SIDE,
            _round_half_up_by_factor(
                width / height * DEFAULT_EDIT_LONG_SIDE,
                RESOLUTION_ALIGNMENT,
            ),
        )
        resized_height = DEFAULT_EDIT_LONG_SIDE
    return resized_width, resized_height


def is_sensenova_u1_model(model_path: str) -> bool:
    """Identify SenseNova-U1 Hub IDs and local base checkpoints."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path) as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(config, dict):
            return False
        architectures = config.get("architectures", [])
        return (
            config.get("model_type") == "neo_chat"
            and isinstance(architectures, list)
            and "NEOChatModel" in architectures
        )

    return model_path.rstrip("/").lower() in SENSENOVA_U1_MODEL_IDS


def is_sensenova_u1_adapter_only_model(model_path: str) -> bool:
    return model_path.rstrip("/").lower() in SENSENOVA_U1_ADAPTER_ONLY_MODEL_IDS
