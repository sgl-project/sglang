# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BaseRealtimeState:
    def dispose(self):
        pass


class RealtimeSession:
    def __init__(self):
        # Store independent per-session state objects by type.
        self._states: dict[type[BaseRealtimeState], BaseRealtimeState] = {}

    def get_or_create_state(
        self, state_cls: type[BaseRealtimeState]
    ) -> BaseRealtimeState:
        state = self._states.get(state_cls)
        if state is None:
            state = state_cls()
            self._states[state_cls] = state
        return state

    def get_state(self, state_cls: type[BaseRealtimeState]) -> BaseRealtimeState | None:
        return self._states.get(state_cls)

    def dispose(self):
        for state in self._states.values():
            state.dispose()
        self._states.clear()


class RealtimeSessionCache:
    def __init__(self, max_sessions: int = 64):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, RealtimeSession] = OrderedDict()

    @staticmethod
    def _resolve_session_id(req: Any) -> str | None:
        session_id = getattr(req, "realtime_session_id", None)
        if isinstance(session_id, str) and session_id:
            return session_id
        return None

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
        released = session is not None
        self._dispose_session(session_id, session)
        logger.info(
            "Realtime session release: session_id=%s released=%s",
            session_id,
            released,
        )
        return released

    def attach(self, req: Any) -> None:
        session_id = self._resolve_session_id(req)
        if session_id is None:
            return

        block_idx = getattr(req, "block_idx", 0) or 0
        if session_id not in self._sessions:
            if block_idx > 0:
                raise ValueError(
                    "Missing realtime session state for "
                    f"session_id={session_id} block_idx={block_idx}."
                )
            self._sessions[session_id] = req.session or RealtimeSession()
        elif block_idx == 0:
            old_session = self._sessions[session_id]
            new_session = req.session or RealtimeSession()
            if old_session is not new_session:
                self._dispose_session(session_id, old_session)
            self._sessions[session_id] = new_session
            logger.info(
                "Realtime session reset: session_id=%s",
                session_id,
            )

        req.session = self._sessions[session_id]
        self._sessions.move_to_end(session_id)
        self._evict_stale_sessions()

    def _evict_stale_sessions(self) -> None:
        while len(self._sessions) > self.max_sessions:
            stale_session_id, stale_session = self._sessions.popitem(last=False)
            self._dispose_session(stale_session_id, stale_session)
            logger.debug("Evicted stale realtime session cache: %s", stale_session_id)
