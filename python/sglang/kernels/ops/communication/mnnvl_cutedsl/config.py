# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration and token-count routing for the MNNVL CuTe DSL backend."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch

__all__ = [
    "KernelTarget",
    "MNNVLCuteDSLConfig",
    "MRangeDispatch",
    "ProtocolKind",
    "StaticProfile",
]


class ProtocolKind(Enum):
    LL = "ll"
    BT = "bt"
    HT = "ht"


PresetT = TypeVar("PresetT")


@dataclass(frozen=True, slots=True)
class KernelTarget(Generic[PresetT]):
    protocol: ProtocolKind
    preset: PresetT


TargetT = TypeVar("TargetT")


@dataclass(frozen=True, slots=True)
class MRangeDispatch(Generic[TargetT]):
    """Map contiguous positive token-count ranges to kernel targets."""

    upper_bounds: tuple[int | None, ...]
    targets: tuple[TargetT, ...]

    def __post_init__(self) -> None:
        if not self.upper_bounds:
            raise ValueError("M range dispatch must contain at least one range")
        if len(self.upper_bounds) != len(self.targets):
            raise ValueError("M range upper bounds and targets must have equal length")

        previous = 0
        for index, upper_bound in enumerate(self.upper_bounds):
            if upper_bound is None:
                if index != len(self.upper_bounds) - 1:
                    raise ValueError("An unbounded M range must be the final range")
                continue
            if upper_bound <= previous:
                raise ValueError("M range upper bounds must be strictly increasing")
            previous = upper_bound

    @property
    def is_unbounded(self) -> bool:
        return self.upper_bounds[-1] is None

    @property
    def finite_upper_bound(self) -> int | None:
        return None if self.is_unbounded else self.upper_bounds[-1]

    def supports(self, m: int) -> bool:
        if m <= 0:
            return False
        upper_bound = self.finite_upper_bound
        return upper_bound is None or m <= upper_bound

    def select(self, m: int) -> TargetT:
        if not self.supports(m):
            raise ValueError(f"No kernel route supports M={m}")

        finite_bounds = tuple(
            upper_bound for upper_bound in self.upper_bounds if upper_bound is not None
        )
        index = bisect_left(finite_bounds, m)
        return self.targets[index]

    def referenced_protocols(self) -> frozenset[ProtocolKind]:
        protocols = {
            target.protocol
            for target in self.targets
            if isinstance(target, KernelTarget)
        }
        return frozenset(protocols)

    def targets_for_capacity(self, capacity_m: int) -> tuple[TargetT, ...]:
        if capacity_m <= 0:
            return ()
        selected = []
        lower_bound = 1
        for upper_bound, target in zip(self.upper_bounds, self.targets, strict=True):
            if lower_bound > capacity_m:
                break
            selected.append(target)
            if upper_bound is None:
                break
            lower_bound = upper_bound + 1
        return tuple(selected)

    def max_m_for_protocol(
        self, protocol: ProtocolKind, *, capacity_m: int
    ) -> int | None:
        lower_bound = 1
        maximum = None
        for upper_bound, target in zip(self.upper_bounds, self.targets, strict=True):
            effective_upper_bound = capacity_m if upper_bound is None else upper_bound
            if (
                isinstance(target, KernelTarget)
                and target.protocol is protocol
                and lower_bound <= capacity_m
            ):
                maximum = min(effective_upper_bound, capacity_m)
            lower_bound = effective_upper_bound + 1
        return maximum


@dataclass(frozen=True, slots=True)
class StaticProfile:
    tp_size: int
    hidden_size: int
    top_k: int
    dtype: torch.dtype
    finalize_routes: MRangeDispatch[KernelTarget[object]]
    all_reduce_routes: MRangeDispatch[KernelTarget[object]]

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % 8:
            raise ValueError("hidden_size must be a positive multiple of 8")

    def matches(
        self,
        *,
        tp_size: int,
        hidden_size: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> bool:
        return (
            self.tp_size == tp_size
            and self.hidden_size == hidden_size
            and self.top_k == top_k
            and self.dtype == dtype
        )

    def validate_capacity(self, capacity_m: int) -> None:
        if capacity_m <= 0:
            raise ValueError("capacity_m must be positive")
        if not self.finalize_routes.supports(capacity_m):
            raise ValueError(
                "Finalize routes do not cover the requested workspace capacity"
            )
        if not self.all_reduce_routes.supports(capacity_m):
            raise ValueError(
                "AllReduce routes do not cover the requested workspace capacity"
            )

    @property
    def referenced_protocols(self) -> frozenset[ProtocolKind]:
        return (
            self.finalize_routes.referenced_protocols()
            | self.all_reduce_routes.referenced_protocols()
        )

    def protocol_capacity(
        self, protocol: ProtocolKind, *, capacity_m: int
    ) -> int | None:
        maxima = (
            self.finalize_routes.max_m_for_protocol(protocol, capacity_m=capacity_m),
            self.all_reduce_routes.max_m_for_protocol(protocol, capacity_m=capacity_m),
        )
        present = tuple(value for value in maxima if value is not None)
        return max(present) if present else None


@dataclass(frozen=True, slots=True)
class MNNVLCuteDSLConfig:
    """Static profiles and routing policy for one backend configuration."""

    profiles: tuple[StaticProfile, ...]

    def __post_init__(self) -> None:
        keys = [
            (profile.tp_size, profile.hidden_size, profile.top_k, profile.dtype)
            for profile in self.profiles
        ]
        if not keys:
            raise ValueError("A backend config must contain at least one profile")
        if len(keys) != len(set(keys)):
            raise ValueError("Backend config profiles must have unique static shapes")

    def resolve(
        self,
        *,
        tp_size: int,
        hidden_size: int,
        top_k: int,
        dtype: torch.dtype,
        capacity_m: int,
    ) -> StaticProfile:
        for profile in self.profiles:
            if profile.matches(
                tp_size=tp_size,
                hidden_size=hidden_size,
                top_k=top_k,
                dtype=dtype,
            ):
                profile.validate_capacity(capacity_m)
                return profile
        raise ValueError("No MNNVL CuTe DSL profile supports this static shape")
