# SPDX-License-Identifier: Apache-2.0
"""Lightweight model constraints shared by expert-pack startup and loading."""

from __future__ import annotations

from typing import Any

DEEPSEEK_V4_MODEL_TYPE = "deepseek_v4"
KIMI_K3_MODEL_TYPE = "kimi_linear"

KIMI_K3_REQUIRED_CONFIG = {
    "num_hidden_layers": 93,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "first_k_dense_replace": 1,
    "routed_expert_hidden_size": 3584,
    "moe_intermediate_size": 3072,
    "num_shared_experts": 2,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
}


def validate_expert_pack_model_config(hf_config: Any) -> tuple[str | None, list[str]]:
    """Return the supported model kind and every violated hard constraint."""
    model_type = getattr(hf_config, "model_type", None)
    if model_type == DEEPSEEK_V4_MODEL_TYPE:
        return DEEPSEEK_V4_MODEL_TYPE, []
    if model_type != KIMI_K3_MODEL_TYPE:
        return None, [
            "model_type must be 'deepseek_v4' or the text-only Kimi-K3 "
            f"'kimi_linear' config, got {model_type!r}"
        ]

    errors = []
    for field, expected in KIMI_K3_REQUIRED_CONFIG.items():
        actual = getattr(hf_config, field, None)
        if actual != expected:
            errors.append(f"Kimi-K3 {field} must be {expected!r}, got {actual!r}")
    return KIMI_K3_MODEL_TYPE, errors
