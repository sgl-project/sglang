# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import RolloutMetadata, RolloutRequest


@dataclass
class AdapterResult:
    output: Any
    rollout_metadata: RolloutMetadata | None = None


class DiffusionRolloutAdapter(Protocol):
    name: str

    def can_handle(self, pipe: Any, request: RolloutRequest) -> bool: ...

    def run(
        self,
        pipe: Any,
        batch: Any,
        kwargs: dict[str, Any],
        request: RolloutRequest,
    ) -> AdapterResult: ...
