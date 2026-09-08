"""Ordering of session lifecycle requests against the simulated clock.

SGLang has no counterpart to mirror: opens and closes are plain IPC structs the
runtime executes on receipt, because on a real server receipt order is execution
order. The two diverge only under a simulated clock.

This module holds no SGLang types and no process-global state. The caller
classifies requests and supplies the completion predicate, so the ordering rules
can be exercised without a scheduler, a request store, or a clock.
"""

from typing import Callable

# Sorts an open ahead of a turn sharing its arrival timestamp; real salts are
# `time.time_ns()` and always positive.
OPEN_SALT = -1


class SessionTimeline:
    """Decides when a held session open or close becomes releasable.

    The caller owns the arrival queue and reports what it knows about pending
    turns; this class never inspects the queue itself.
    """

    def __init__(self, is_request_finished: Callable[[str], bool]):
        self._is_request_finished = is_request_finished
        self._pending_opens: list[tuple[str, object]] = []
        self._pending_closes: list[tuple[str, object]] = []
        self._session_rids: dict[str, set[str]] = {}

    def reset(self) -> None:
        self._pending_opens.clear()
        self._pending_closes.clear()
        self._session_rids.clear()

    def hold_open(self, *, session_id: str, req: object) -> None:
        self._pending_opens.append((session_id, req))

    def hold_close(self, *, session_id: str, req: object) -> None:
        self._pending_closes.append((session_id, req))

    def has_pending_opens(self) -> bool:
        return len(self._pending_opens) > 0

    def take_opens(
        self, first_arrival: dict[str, float]
    ) -> tuple[list[tuple[float, object]], list[object]]:
        """Drain held opens into (timestamped, releasable-now) partitions.

        An open whose session has no arrival left to precede is releasable rather
        than stranded, which would leave the session unopened for its turns.
        """
        timestamped = []
        releasable = []
        for session_id, req in self._pending_opens:
            arrival = first_arrival.get(session_id)
            if arrival is None:
                releasable.append(req)
            else:
                timestamped.append((arrival, req))
        self._pending_opens.clear()
        return timestamped, releasable

    def note_dispatched(self, *, session_id: str, rid: str) -> None:
        """Attribute a dispatched turn to its session.

        Must run before `take_settled_closes` in the same iteration, or a turn
        dispatched now is not yet counted and its session reads as settled.
        """
        self._session_rids.setdefault(session_id, set()).add(rid)

    def take_settled_closes(
        self, sessions_with_pending_turns: set[str]
    ) -> list[object]:
        """Release closes whose session has no turn left to dispatch or finish."""
        released = []
        still_pending = []
        for session_id, req in self._pending_closes:
            if self._is_settled(session_id, sessions_with_pending_turns):
                released.append(req)
            else:
                still_pending.append((session_id, req))
        self._pending_closes = still_pending
        return released

    def _is_settled(
        self, session_id: str, sessions_with_pending_turns: set[str]
    ) -> bool:
        if session_id in sessions_with_pending_turns:
            return False
        return all(
            self._is_request_finished(rid)
            for rid in self._session_rids.get(session_id, ())
        )
