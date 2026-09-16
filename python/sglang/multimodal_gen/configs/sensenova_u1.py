# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the native SenseNova-U1 integration."""

import json
import os

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
SENSENOVA_U1_RESOLUTION_ALIGNMENT = 32

DEFAULT_CFG_NORM = "none"
DEFAULT_TIMESTEP_SHIFT = 3.0
DEFAULT_ENABLE_TIMESTEP_SHIFT = True
DEFAULT_CFG_INTERVAL = (0.0, 1.0)
DEFAULT_T_EPS = 0.02
DEFAULT_THINK_MODE = False


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
