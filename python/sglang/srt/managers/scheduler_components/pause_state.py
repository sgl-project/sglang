from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional, Tuple

from sglang.srt.managers.io_struct import PauseGenerationReqInput

# Pause windows kept for prefill busy accounting.
PAUSE_WINDOWS_MAX = 1024


class PauseState:
    """The state of the engine's pause: whether it is paused (`pause_generation`
    until `continue_generation`), the inputs held behind a pause until the loop
    reaches a point where it can apply it, and the pause windows that step timing
    leaves out. The event loops ask it whether they may launch."""

    def __init__(self) -> None:
        self._paused = False
        # Received inputs not yet dispatched, in order: a held pause and all after it.
        self._held_inputs: List = []
        # Closed pause windows a later prefill step may still overlap; not busy time.
        self._windows: Deque[Tuple[float, float]] = deque(maxlen=PAUSE_WINDOWS_MAX)
        self._window_start: Optional[float] = None

    @property
    def paused(self) -> bool:
        return self._paused

    def holds_inputs(self) -> bool:
        return bool(self._held_inputs)

    def may_launch(self) -> bool:
        """No new batch while paused or while a pause is held."""
        return not (self._paused or self._held_inputs)

    def take_inputs(self, received: List, *, stop_at_pause: bool) -> List:
        """Queue `received` after the held inputs and return the ones to dispatch
        now: all of them, or with `stop_at_pause` those before the first pause,
        which stays held with everything after it."""
        self._held_inputs.extend(received)
        num_to_dispatch = len(self._held_inputs)
        if stop_at_pause:
            for index, req in enumerate(self._held_inputs):
                if isinstance(req, PauseGenerationReqInput):
                    num_to_dispatch = index
                    break
        to_dispatch = self._held_inputs[:num_to_dispatch]
        del self._held_inputs[:num_to_dispatch]
        return to_dispatch

    def pause(self) -> None:
        self._paused = True

    def start_window(self) -> None:
        """Start the pause window, unless one is open: an in-place pause starts it
        at once, a retract once the in-flight result has drained."""
        if self._window_start is None:
            self._window_start = time.monotonic()

    def resume(self) -> None:
        self._paused = False
        if self._window_start is not None:
            self._windows.append((self._window_start, time.monotonic()))
            self._window_start = None

    def consume_paused_time(self, start_ts: float, end_ts: float) -> float:
        """Seconds of [start_ts, end_ts] spent paused, the open window included.
        Drops the closed windows that end by `end_ts`: later prefill steps start at
        or after it."""
        windows = list(self._windows)
        if self._window_start is not None:
            # A result processed while still paused excludes the pause so far.
            windows.append((self._window_start, end_ts))
        paused = sum(max(0.0, min(end_ts, b) - max(start_ts, a)) for a, b in windows)
        while self._windows and self._windows[0][1] <= end_ts:
            self._windows.popleft()
        return paused
