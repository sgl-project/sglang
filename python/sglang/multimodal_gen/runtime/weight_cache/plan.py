# SPDX-License-Identifier: Apache-2.0
"""Immutable compatibility and execution descriptions; no live process groups."""

import dataclasses
import json
from dataclasses import dataclass

from sglang.weight_cache_common.descriptors import CACHE_ABI, canonical_digest


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PlannedRankContext:
    rank: int
    local_device: int
    device_uuid: str
    tp_size: int = 1
    tp_rank: int = 0


@dataclass(frozen=True)
class ComponentPlan:
    name: str
    library: str
    architecture: str
    source_path: str
    residency: str
    attention: str | None
    cached: bool
    loader_id: str


@dataclass(frozen=True)
class PipelineExecutionPlan:
    pipeline: str
    components: tuple[ComponentPlan, ...]

    @property
    def digest(self):
        return canonical_digest(dataclasses.asdict(self))


@dataclass(frozen=True)
class CacheCompatibilityPlan:
    # JSON bytes (rather than a mutable nested dict) are the authoritative plan.
    document: str

    @property
    def digest(self):
        return canonical_digest(self.to_dict())

    def to_dict(self):
        return json.loads(self.document)

    @classmethod
    def from_fields(cls, **fields):
        return cls(
            canonical_json(
                dict(
                    cache_abi=CACHE_ABI,
                    family="diffusion",
                    protocol_version=1,
                    **fields,
                )
            )
        )


def plan_diff(expected, actual, prefix="") -> dict:
    result = {}
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            result.update(
                plan_diff(
                    expected.get(key),
                    actual.get(key),
                    f"{prefix}.{key}" if prefix else key,
                )
            )
    elif expected != actual:
        result[prefix] = {"expected": expected, "actual": actual}
    return result
