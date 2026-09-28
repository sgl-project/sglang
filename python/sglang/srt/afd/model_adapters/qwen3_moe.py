"""Qwen3-MoE family identity for AFD."""

from __future__ import annotations

from typing import Any

from ..metadata import FlashAttentionMetadataGuard
from .base import AFDDecoderAdapter

# One AFD path serves every Qwen3MoeForCausalLM checkpoint, so the gate is an
# explicit set of validated shapes rather than a single parameter count.
_QWEN3_MOE_IDENTITIES = frozenset(
    {
        # Qwen3-30B-A3B
        (48, 2048, 128, 8),
        # Qwen3-235B-A22B
        (94, 4096, 128, 8),
    }
)


def matches_qwen3_moe(model: Any) -> bool:
    """Return whether a model has an admitted C1 Qwen3-MoE identity."""

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
        and actual in _QWEN3_MOE_IDENTITIES
    )


class Qwen3AFDAdapter(AFDDecoderAdapter):
    guard_class = FlashAttentionMetadataGuard
    family_error = "AFD_QWEN3_MOE_REQUIRED"

    def matches_family(self) -> bool:
        return matches_qwen3_moe(self.model)
