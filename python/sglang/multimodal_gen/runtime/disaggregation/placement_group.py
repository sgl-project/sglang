# SPDX-License-Identifier: Apache-2.0
"""Declarative placement-group configuration for N-group disaggregation.

A PlacementGroupConfig describes which pipeline stages colocate on the
same GPU pool.  The existing 3-role (encoder/denoiser/decoder) topology
is one specific configuration; arbitrary N-group topologies are supported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any


@dataclass
class PlacementGroupSpec:
    """One disaggregation group: a named set of colocated pipeline stages."""

    name: str
    stage_patterns: list[str] = field(default_factory=list)
    modules: list[str] | None = None

    def matches_stage(self, stage_name: str) -> bool:
        return any(fnmatch(stage_name, pat) for pat in self.stage_patterns)


@dataclass
class PlacementGroupConfig:
    """Complete disaggregation topology as an ordered chain of groups.

    Groups are ordered: the first group receives raw requests, the last
    group returns final output.  Intermediate groups receive tensors via
    the transfer protocol and forward to the next group.
    """

    groups: list[PlacementGroupSpec] = field(default_factory=list)

    def group_chain(self) -> list[str]:
        return [g.name for g in self.groups]

    def next_group(self, name: str) -> str | None:
        chain = self.group_chain()
        try:
            idx = chain.index(name)
        except ValueError:
            return None
        if idx + 1 < len(chain):
            return chain[idx + 1]
        return None

    def prev_group(self, name: str) -> str | None:
        chain = self.group_chain()
        try:
            idx = chain.index(name)
        except ValueError:
            return None
        if idx > 0:
            return chain[idx - 1]
        return None

    def group_index(self, name: str) -> int:
        chain = self.group_chain()
        return chain.index(name)

    def is_first(self, name: str) -> bool:
        chain = self.group_chain()
        return len(chain) > 0 and chain[0] == name

    def is_last(self, name: str) -> bool:
        chain = self.group_chain()
        return len(chain) > 0 and chain[-1] == name

    def get_group(self, name: str) -> PlacementGroupSpec | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def stage_belongs_to_group(self, stage_name: str, group_name: str) -> bool:
        group = self.get_group(group_name)
        if group is None:
            return False
        return group.matches_stage(stage_name)

    def get_group_modules(self, group_name: str) -> list[str] | None:
        group = self.get_group(group_name)
        if group is None:
            return None
        return group.modules

    def validate(self, stage_names: list[str] | None = None) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors: list[str] = []
        if not self.groups:
            errors.append("PlacementGroupConfig must have at least one group")
            return errors

        names = [g.name for g in self.groups]
        if len(names) != len(set(names)):
            errors.append(f"Duplicate group names: {names}")

        if stage_names is not None:
            for sn in stage_names:
                matched = [g.name for g in self.groups if g.matches_stage(sn)]
                if not matched:
                    errors.append(f"Stage '{sn}' not matched by any group")
                elif len(matched) > 1:
                    errors.append(f"Stage '{sn}' matched by multiple groups: {matched}")
        return errors

    @classmethod
    def classic_3_role(cls) -> PlacementGroupConfig:
        return cls(
            groups=[
                PlacementGroupSpec(
                    name="encoder",
                    stage_patterns=["*"],
                    modules=[
                        "text_encoder",
                        "tokenizer",
                        "image_encoder",
                        "image_processor",
                        "processor",
                        "connectors",
                        "vae",
                        "audio_vae",
                        "video_vae",
                    ],
                ),
                PlacementGroupSpec(
                    name="denoiser",
                    stage_patterns=["*Denoising*", "*Refinement*"],
                    modules=["transformer", "scheduler", "vae", "audio_vae"],
                ),
                PlacementGroupSpec(
                    name="decoder",
                    stage_patterns=["*Decoding*"],
                    modules=["vae", "audio_vae", "video_vae", "vocoder"],
                ),
            ]
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlacementGroupConfig:
        groups = []
        for g in data.get("groups", []):
            groups.append(
                PlacementGroupSpec(
                    name=g["name"],
                    stage_patterns=g.get("stage_patterns", []),
                    modules=g.get("modules"),
                )
            )
        return cls(groups=groups)

    @classmethod
    def from_json(cls, json_str: str) -> PlacementGroupConfig:
        if json_str.startswith("@"):
            filepath = json_str[1:]
            with open(filepath) as f:
                data = json.load(f)
        else:
            data = json.loads(json_str)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "name": g.name,
                    "stage_patterns": g.stage_patterns,
                    **({"modules": g.modules} if g.modules is not None else {}),
                }
                for g in self.groups
            ]
        }
