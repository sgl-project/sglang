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
# Off by default, matching upstream: OpenSenseNova/SenseNova-U1's own
# examples/t2i/inference.py gates PE behind `--enhance` (action="store_true"),
# not a default-on flag.
DEFAULT_USE_PE = False

# The official SenseNova-U1 checkpoint ships no local prompt-enhancement (PE)
# component -- unlike ERNIE-Image's bundled `pe/` subfolder, PE is always a
# remote call to an OpenAI-compatible chat/completions endpoint (mirroring
# OpenSenseNova/SenseNova-U1's own U1_ENHANCE_* contract), configured via the
# generic server_args.pe_backend/pe_endpoint/pe_model_name/pe_api_key fields.
# It only activates when the caller supplies an API key.
DEFAULT_PE_SYSTEM_PROMPT = (
    "You are a prompt engineer for a text-to-image model. Expand the user's "
    "prompt into a detailed, vivid, well-structured English description "
    "suitable for image generation. Preserve the user's intent and any "
    "explicit constraints. Return only the expanded prompt, with no extra "
    "commentary."
)


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
