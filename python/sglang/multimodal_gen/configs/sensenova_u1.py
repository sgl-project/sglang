# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

import json
import os
from enum import Enum

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

DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT = True
DEFAULT_CFG_INTERVAL = (0.0, 1.0)
DEFAULT_IMG_CFG_SCALE = 1.0
DEFAULT_T_EPS = 0.02
DEFAULT_THINK_MODE = False


class SenseNovaGuidanceProfile(Enum):
    """Request-level conditioning branches used while guidance is active."""

    CONDITION = ("condition",)
    CONDITION_IMAGE = ("condition", "image_condition")
    CONDITION_UNCONDITIONAL = ("condition", "uncondition")
    CONDITION_IMAGE_UNCONDITIONAL = (
        "condition",
        "image_condition",
        "uncondition",
    )

    @property
    def branch_count(self) -> int:
        return len(self.value)

    @property
    def needs_image_condition(self) -> bool:
        return "image_condition" in self.value

    @property
    def needs_uncondition(self) -> bool:
        return "uncondition" in self.value

    @property
    def has_separate_cfg(self) -> bool:
        return self.branch_count == 2


def derive_guidance_profile(
    *, is_edit: bool, cfg_scale: float, img_cfg_scale: float
) -> SenseNovaGuidanceProfile:
    """Derive the branch profile shared by generation and Cache-DiT.

    Exact comparisons intentionally match the native denoising loops. A
    tolerance could select a cache profile that advances residual state on the
    wrong conditioning branch.
    """
    if not is_edit:
        return (
            SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL
            if cfg_scale > 1
            else SenseNovaGuidanceProfile.CONDITION
        )

    if cfg_scale == 1 and img_cfg_scale == 1:
        return SenseNovaGuidanceProfile.CONDITION
    if img_cfg_scale == 1:
        return SenseNovaGuidanceProfile.CONDITION_IMAGE
    if cfg_scale == img_cfg_scale:
        return SenseNovaGuidanceProfile.CONDITION_UNCONDITIONAL
    return SenseNovaGuidanceProfile.CONDITION_IMAGE_UNCONDITIONAL


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
