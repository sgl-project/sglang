# SPDX-License-Identifier: Apache-2.0
"""Validated logical expert IDs for startup-time physical pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpertProfile:
    num_experts: int
    keep: tuple[tuple[int, ...], ...]

    @property
    def num_kept(self) -> int:
        return len(self.keep[0])

    def validate_model(self, num_layers: int, num_experts: int, top_k: int) -> None:
        if self.num_experts != num_experts:
            raise ValueError("Expert profile num_experts does not match the model")
        if len(self.keep) != num_layers:
            raise ValueError("Expert profile must contain every model layer")
        if self.num_kept < top_k:
            raise ValueError("Expert profile must retain at least num_experts_per_tok")

    def to_json(self) -> str:
        return json.dumps({"num_experts": self.num_experts, "keep": self.keep})


def load_expert_profile(source: str) -> ExpertProfile:
    """Read a JSON object or path. Resolvers serialize it before spawning workers."""
    contents = source if source.lstrip().startswith("{") else Path(source).read_text()
    value = json.loads(contents)
    if not isinstance(value, dict) or set(value) != {"num_experts", "keep"}:
        raise ValueError("Expert profile requires exactly num_experts and keep")
    num_experts = value["num_experts"]
    keep = value["keep"]
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("Expert profile num_experts must be a positive integer")
    if not isinstance(keep, list) or not keep:
        raise ValueError("Expert profile keep must be a nonempty list of layers")
    for layer_id, ids in enumerate(keep):
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Expert profile layer {layer_id} must be a nonempty list")
        if any(type(i) is not int or not 0 <= i < num_experts for i in ids):
            raise ValueError(f"Expert profile layer {layer_id} contains invalid IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Expert profile layer {layer_id} contains duplicate IDs")
        if len(ids) != len(keep[0]):
            raise ValueError("Expert profile must retain equal capacity in every layer")
    return ExpertProfile(num_experts, tuple(tuple(ids) for ids in keep))


def get_num_physical_experts(num_logical_experts: int) -> int:
    from sglang.srt.runtime_context import get_exec

    source = get_exec().moe.moe_expert_profile
    if source is not None:
        return load_expert_profile(source).num_kept
    return num_logical_experts + get_exec().moe.ep_num_redundant_experts
