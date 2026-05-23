from __future__ import annotations

from enum import IntEnum

import torch


def _phase_repr(phase: int | IntEnum) -> str:
    if isinstance(phase, IntEnum):
        return f"{phase.name}({int(phase)})"
    return str(int(phase))


class SimplePhaseChecker:
    """GPU-side state machine for any int-keyed phase sequence."""

    def __init__(self, *, initial_phase: int | IntEnum, device: torch.device) -> None:
        self._phase = torch.tensor(int(initial_phase), dtype=torch.int32, device=device)
        print(
            f"[SimplePhaseChecker.__init__] device={device} "
            f"initial_phase={_phase_repr(initial_phase)}",
            flush=True,
        )

    def update(
        self,
        *,
        expect_phase: int | IntEnum,
        next_phase: int | IntEnum,
        caller_name: str = "",
    ) -> None:
        print(
            f"[SimplePhaseChecker.update] caller={caller_name!r} "
            f"expect={_phase_repr(expect_phase)} "
            f"next={_phase_repr(next_phase)} "
            f"capturing={torch.cuda.is_current_stream_capturing()}",
            flush=True,
        )
        torch._assert_async(
            self._phase == int(expect_phase),
            f"SimplePhaseChecker: phase mismatch in {caller_name!r} — "
            f"expected {_phase_repr(expect_phase)} before transitioning to "
            f"{_phase_repr(next_phase)}",
        )
        self._phase.fill_(int(next_phase))
