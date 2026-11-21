from __future__ import annotations

from typing import Any, List, Optional, Tuple

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    SequencedControlReq,
)


class ControlSeqGate:
    """Holds each numbered control request (`SequencedControlReq`) until every
    attention-DP group has received it. Each group receives it in its own step,
    and the MLP-sync gather each group runs every step reports the highest
    number each has received (`release`). Passing a request on once that gather
    shows every group holds it makes all groups apply it between the same two
    gathers, so its collectives meet. A paused engine runs no MLP-sync gather
    for those collectives to cross, so while paused, numbered requests pass as
    they arrive.

    Each sender hands a group its numbered requests in number order, so the
    highest number received stands for every one before it that was sent to
    this group; a group the DP controller skipped while it was inactive, or one
    that joined later, goes on from the numbers it is sent."""

    def __init__(self) -> None:
        # Numbered requests not yet passed on, in order.
        self._held: List[Tuple[int, Any]] = []
        # The highest number this group received, and the number every
        # attention-DP group had received up to at the last MLP-sync gather.
        self._seq_received = 0
        self._seq_released = 0
        # Whether the engine is paused once the inputs passed on are applied:
        # the pause and continue handlers always pause and resume.
        self._paused = False

    @property
    def seq_received(self) -> int:
        return self._seq_received

    def release(self, seq: int) -> None:
        """Every attention-DP group has received the requests numbered up to seq."""
        self._seq_released = max(self._seq_released, seq)

    def holds_inputs(self) -> bool:
        """A numbered request is held: the next gathers may release it."""
        return bool(self._held)

    def take_inputs(self, received: List, *, align: bool) -> List:
        """Queue `received` after the held requests and return the inputs to pass
        on, in order, numbered ones unwrapped. With `align` (the attention-DP
        groups gather every step), a numbered request that not every group has
        received yet stays held while not paused, with the numbered requests
        after it; unnumbered inputs pass."""
        pending: List[Tuple[Optional[int], Any]] = list(self._held)
        for req in received:
            if isinstance(req, SequencedControlReq):
                self._seq_received = max(self._seq_received, req.seq)
                pending.append((req.seq, req.unwrap()))
            else:
                pending.append((None, req))

        awaiting_seq = False
        self._held = []
        to_pass = []
        for seq, req in pending:
            if seq is not None:
                awaiting_seq = awaiting_seq or (
                    align and not self._paused and seq > self._seq_released
                )
                if awaiting_seq:
                    self._held.append((seq, req))
                    continue
            if isinstance(req, PauseGenerationReqInput):
                self._paused = True
            elif isinstance(req, ContinueGenerationReqInput):
                self._paused = False
            to_pass.append(req)
        return to_pass
