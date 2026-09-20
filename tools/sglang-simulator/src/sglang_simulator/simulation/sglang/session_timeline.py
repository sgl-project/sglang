"""Ordering of session lifecycle requests against the simulated clock.

SGLang has no counterpart to mirror: opens and closes are plain IPC structs the
runtime executes on receipt, because on a real server receipt order is execution
order. The two diverge only under a simulated clock.

This module holds no SGLang types and no process-global state. The caller
classifies requests and supplies the completion predicate, so the ordering rules
can be exercised without a scheduler, a request store, or a clock.
"""

from typing import Callable


class SessionTimeline:
    """Decides when a held session close becomes releasable.

    Only closes are held. `open_session` blocks on a scheduler response
    (`tokenizer_control_mixin.py`), so a client awaits it before submitting the
    session's turns -- delaying an open until those turns arrive deadlocks.
    A close is fire-and-forget, and is the one that frees KV early.
    """

    def __init__(self, is_request_finished: Callable[[str], bool]):
        self._is_request_finished = is_request_finished
        self._pending_closes: list[tuple[str, object]] = []
        self._session_rids: dict[str, set[str]] = {}

    def reset(self) -> None:
        self._pending_closes.clear()
        self._session_rids.clear()

    def has_pending(self) -> bool:
        """Whether any close is held. Lets callers skip the settle scan."""
        return bool(self._pending_closes)

    def pending_session_ids(self) -> list[str]:
        """Sessions whose close is still held, for drain and leak reporting."""
        return [session_id for session_id, _ in self._pending_closes]

    def drain(self) -> list[object]:
        """Release every held close regardless of settlement.

        A turn that never finishes -- aborted, retired, or failed mid-decode --
        leaves its session permanently unsettled, because the completion
        predicate counts generated tokens against `max_new_tokens`. Without a
        drain those closes are never delivered and the session's KV stays
        referenced for the life of the process.
        """
        released = [req for _, req in self._pending_closes]
        self._pending_closes = []
        return released

    def hold_close(self, *, session_id: str, req: object) -> None:
        self._pending_closes.append((session_id, req))

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
        rids = self._session_rids.get(session_id)
        if not rids:
            # No turn of this session has been dispatched yet. `all()` over an
            # empty set is vacuously true, which would release the close on the
            # spot -- the exact reordering this class exists to prevent. A
            # client that pipelines its close alongside the final turn hits this
            # race. Stay unsettled until a turn is seen; `drain()` covers the
            # session that legitimately never sends one.
            return False
        return all(self._is_request_finished(rid) for rid in rids)
