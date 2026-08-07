# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RealtimeSessionCapacityError(RuntimeError):
    pass


class BaseRealtimeState:
    """per-session state owned by pipeline stages"""

    def dispose(self) -> None:
        pass


class RealtimeSession:
    """reusable state container across realtime request chunks"""

    def __init__(self) -> None:
        self._states: dict[type[BaseRealtimeState], BaseRealtimeState] = {}

    @staticmethod
    def resolve_session_id(req: Any) -> str | None:
        session_id = req.realtime_session_id
        if isinstance(session_id, str) and session_id:
            return session_id
        return None

    def get_or_create_state(
        self, state_cls: type[BaseRealtimeState]
    ) -> BaseRealtimeState:
        """returns the BaseRealtimeState instance hold by the current RealtimeSession"""
        state = self._states.get(state_cls)
        if state is None:
            state = state_cls()
            self._states[state_cls] = state
        return state

    def get_state(self, state_cls: type[BaseRealtimeState]) -> BaseRealtimeState | None:
        return self._states.get(state_cls)

    def dispose(self) -> None:
        for state in list(self._states.values()):
            state.dispose()
        self._states.clear()


class RealtimeSessionCache:
    """Binds incoming chunks to persistent realtime sessions without eviction."""

    def __init__(
        self,
        max_sessions: int = 64,
        *,
        stale_after_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_sessions = max_sessions
        self.stale_after_s = stale_after_s
        self._clock = clock
        self._sessions: OrderedDict[str, RealtimeSession] = OrderedDict()
        self._last_seen: dict[str, float] = {}

    def _dispose_session(
        self, session_id: str, session: RealtimeSession | None
    ) -> None:
        if session is None:
            return
        try:
            session.dispose()
        except Exception as e:
            logger.warning(
                "Failed to dispose realtime session cache entry %s: %s",
                session_id,
                e,
            )

    def release(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        self._last_seen.pop(session_id, None)
        released = session is not None
        self._dispose_session(session_id, session)
        logger.info(
            "Realtime session release: session_id=%s released=%s",
            session_id,
            released,
        )
        return released

    def _prune_stale(self, now: float) -> None:
        if self.stale_after_s is None:
            return
        stale = [
            session_id
            for session_id, last_seen in self._last_seen.items()
            if now - last_seen >= self.stale_after_s
        ]
        for session_id in stale:
            session = self._sessions.pop(session_id, None)
            self._last_seen.pop(session_id, None)
            self._dispose_session(session_id, session)
            logger.warning(
                "Reclaimed stale realtime session cache entry: session_id=%s",
                session_id,
            )

    def attach(self, req: Any) -> None:
        session_id = RealtimeSession.resolve_session_id(req)
        if session_id is None:
            return

        now = self._clock()
        self._prune_stale(now)

        if session_id not in self._sessions:
            if req.block_idx > 0:
                raise ValueError(
                    "Missing realtime session state for "
                    f"session_id={session_id} block_idx={req.block_idx}."
                )
            if len(self._sessions) >= self.max_sessions:
                raise RealtimeSessionCapacityError(
                    "Realtime session capacity exhausted: "
                    f"active={len(self._sessions)} max={self.max_sessions}"
                )
            self._sessions[session_id] = req.session or RealtimeSession()
        elif req.block_idx == 0:
            old_session = self._sessions[session_id]
            new_session = req.session or RealtimeSession()
            if old_session is not new_session:
                self._dispose_session(session_id, old_session)
            self._sessions[session_id] = new_session
            logger.info("Realtime session reset: session_id=%s", session_id)

        req.session = self._sessions[session_id]
        self._last_seen[session_id] = now
        self._sessions.move_to_end(session_id)
