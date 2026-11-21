from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    SequencedControlReq,
)

# Pause windows kept for prefill busy accounting.
PAUSE_WINDOWS_MAX = 1024


class PauseState:
    """The state of the engine's pause: whether it is paused (`pause_generation`
    until `continue_generation`), the inputs held until the loop can apply them,
    and the pause windows that step timing leaves out. The event loops ask it
    whether they may launch.

    Inputs are held behind a pause until the loop reaches a point where it can
    apply it, and a numbered control request (`SequencedControlReq`) until every
    attention-DP group has received it: each group receives it in its own step,
    and the MLP-sync gather each group runs every step reports the highest
    number each has received (`release_control_seqs`). Applying a request once
    that gather shows every group holds it makes all groups apply it between the
    same two gathers, so its collectives meet. A paused engine runs no MLP-sync
    gather for those collectives to cross, so while paused, numbered requests
    are applied as they arrive.

    Each sender hands a group its numbered requests in number order, so the
    highest number received stands for every one before it that was sent to
    this group; a group the DP controller skipped while it was inactive, or one
    that joined later, goes on from the numbers it is sent."""

    def __init__(self) -> None:
        self._paused = False
        # Received inputs not yet dispatched, in order, each with its control
        # number (None when it has none).
        self._held_inputs: List[Tuple[Optional[int], Any]] = []
        # Whether a pause and everything after it are held (`stop_at_pause`).
        self._holds_pause = False
        # The highest control number this group received, and the number every
        # attention-DP group had received up to at the last MLP-sync gather.
        self._control_seq_received = 0
        self._control_seq_released = 0
        # Closed pause windows a later prefill step may still overlap; not busy time.
        self._windows: Deque[Tuple[float, float]] = deque(maxlen=PAUSE_WINDOWS_MAX)
        self._window_start: Optional[float] = None

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def control_seq_received(self) -> int:
        return self._control_seq_received

    def holds_pause(self) -> bool:
        return self._holds_pause

    def may_launch(self) -> bool:
        """No new batch while paused or while a pause is held."""
        return not (self._paused or self._holds_pause)

    def release_control_seqs(self, seq: int) -> None:
        """Every attention-DP group has received the control requests numbered up to seq."""
        self._control_seq_released = max(self._control_seq_released, seq)

    def holds_control_reqs(self) -> bool:
        """A numbered request is held: the next gathers may release it."""
        return any(seq is not None for seq, _ in self._held_inputs)

    def take_inputs(
        self, received: List, *, stop_at_pause: bool, align_control_reqs: bool
    ) -> List:
        """Queue `received` after the held inputs and return the ones to dispatch
        now, in order. With `stop_at_pause`, a pause and everything after it stay
        held. With `align_control_reqs` (the attention-DP groups gather every
        step), a numbered control request that not every group has received yet
        stays held while not paused, with the numbered requests after it."""
        for req in received:
            if isinstance(req, SequencedControlReq):
                self._control_seq_received = max(self._control_seq_received, req.seq)
                self._held_inputs.append((req.seq, req.unwrap()))
            else:
                self._held_inputs.append((None, req))

        # Whether the engine is paused once the inputs before this one are applied.
        paused = self._paused
        awaiting_seq = False
        self._holds_pause = False
        to_dispatch, still_held = [], []
        for index, (seq, req) in enumerate(self._held_inputs):
            if seq is not None:
                awaiting_seq = awaiting_seq or (
                    align_control_reqs
                    and not paused
                    and seq > self._control_seq_released
                )
                if awaiting_seq:
                    still_held.append((seq, req))
                    continue
            if stop_at_pause and isinstance(req, PauseGenerationReqInput):
                self._holds_pause = True
                still_held += self._held_inputs[index:]
                break
            if isinstance(req, PauseGenerationReqInput):
                paused = True
            elif isinstance(req, ContinueGenerationReqInput):
                paused = False
            to_dispatch.append(req)
        self._held_inputs = still_held
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
