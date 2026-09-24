# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-layout helpers for LLaDA2 models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

_EXPERT_PROJECTION_NAMES = frozenset({"gate_proj", "up_proj", "down_proj"})


def prepare_llada2_language_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    num_experts: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Normalize language-model prefixes and expand fused expert tensors."""
    for name, loaded_weight in weights:
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
        elif name.startswith("model.lm_head."):
            name = "lm_head." + name.removeprefix("model.lm_head.")

        expert_prefix, separator, projection_name = name.rpartition(".")
        is_fused_expert = (
            separator
            and expert_prefix.endswith(".mlp.experts")
            and projection_name in _EXPERT_PROJECTION_NAMES
        )
        if not is_fused_expert:
            yield name, loaded_weight
            continue

        if loaded_weight.ndim != 3 or loaded_weight.shape[0] != num_experts:
            raise ValueError(
                f"Invalid fused expert weight {name!r}: expected first dimension "
                f"{num_experts}, got shape={tuple(loaded_weight.shape)}"
            )
        for expert_id in range(num_experts):
            yield (
                f"{expert_prefix}.{expert_id}.{projection_name}.weight",
                loaded_weight[expert_id],
            )
