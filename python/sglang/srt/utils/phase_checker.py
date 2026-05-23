from __future__ import annotations

import torch


class SimplePhaseChecker:
    """GPU-side state machine for any int-keyed phase sequence.

    The phase value lives in a 1-element int32 tensor on GPU. Every
    :meth:`update` issues two async device ops:

    1. ``torch._assert_async(phase == expect_phase)`` — async assert (same
       pattern as ``torch.cuda.is_current_stream_capturing()``-friendly
       checks like ``spec_utils.maybe_detect_nan``); the error surfaces at
       the next host sync, never blocks, so the check is cuda-graph-safe.
    2. ``phase.fill_(next_phase)`` — single async kernel writing the scalar.
       Captured into the graph and replayed verbatim under cuda graph capture.

    Phases are plain ``int`` values; the caller chooses the encoding (e.g.
    an ``IntEnum``). This class does not know about any specific lifecycle."""

    def __init__(self, *, initial_phase: int, device: torch.device) -> None:
        self._phase = torch.tensor(int(initial_phase), dtype=torch.int32, device=device)

    def update(self, *, expect_phase: int, next_phase: int) -> None:
        torch._assert_async(
            self._phase == int(expect_phase),
            f"SimplePhaseChecker: phase mismatch — expected {int(expect_phase)} "
            f"before transitioning to {int(next_phase)}",
        )
        self._phase.fill_(int(next_phase))
