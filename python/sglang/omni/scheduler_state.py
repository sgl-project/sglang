# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.omni.protocol import OmniContextBundle

if TYPE_CHECKING:
    from sglang.omni.coordinator import OmniCoordinator


@dataclass(slots=True)
class OmniSessionRecord:
    context: OmniContextBundle
    turns: int
    created_at: float
    updated_at: float


@dataclass(slots=True)
class OmniSchedulerState:
    orchestrators: dict[str, "OmniCoordinator"] = field(default_factory=dict)
    sessions: dict[str, OmniSessionRecord] = field(default_factory=dict)
