"""Qwen3-MoE family identity for AFD."""

from __future__ import annotations

from typing import Any

from .base import AFDDecoderAdapter

# Geometry comes from the native configuration; operator/placement admission
# is checked separately by the selected execution backend.


def matches_qwen3_moe(model: Any) -> bool:
    """Return whether a model has an admitted Qwen3-MoE identity."""

    try:
        config = model.model.config
        actual = (
            config.num_hidden_layers,
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok,
        )
    except (AttributeError, TypeError):
        return False
    return (
        type(model).__name__ == "Qwen3MoeForCausalLM"
        and all(type(value) is int and value > 0 for value in actual)
        and config.num_experts_per_tok <= config.num_experts
    )


class Qwen3AFDAdapter(AFDDecoderAdapter):
    family_error = "AFD_QWEN3_MOE_REQUIRED"

    def matches_family(self) -> bool:
        return matches_qwen3_moe(self.model)
