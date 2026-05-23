from __future__ import annotations

from enum import IntEnum

import torch


class SimplePhaseChecker:
    """GPU-side state machine for any int-keyed phase sequence."""

    def __init__(self, *, initial_phase: int | IntEnum, device: torch.device) -> None:
        self._phase = torch.tensor(int(initial_phase), dtype=torch.int32, device=device)

    def update(self, *, expect_phase: int | IntEnum, next_phase: int | IntEnum) -> None:
        torch._assert_async(
            self._phase == int(expect_phase),
            f"SimplePhaseChecker: phase mismatch — expected {int(expect_phase)} "
            f"before transitioning to {int(next_phase)}",
        )
        self._phase.fill_(int(next_phase))
