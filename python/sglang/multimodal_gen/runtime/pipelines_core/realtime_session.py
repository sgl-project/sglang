from __future__ import annotations

from collections import OrderedDict
from typing import Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BaseRealtimeState:
    def __init__(self):
        self.kv_cache: Any = None
        self.crossattn_cache: Any = None

    def dispose(self):
        self.kv_cache = None
        self.crossattn_cache = None


class RealtimeSession:
    def __init__(self):
        self.state: BaseRealtimeState | None = None

    @staticmethod
    def resolve_session_id(req: Any) -> str | None:
        has_explicit_session_id = (
            isinstance(req.extra, dict) and "realtime_session_id" in req.extra
        )
        if not has_explicit_session_id and req.session is None:
            return None

        session_id = None
        if has_explicit_session_id:
            session_id = req.extra.get("realtime_session_id")
        elif isinstance(req.request_id, str) and "_" in req.request_id:
            # Backward compatibility for callers that did not set realtime_session_id.
            session_id = req.request_id.split("_", 1)[0]

        if isinstance(session_id, str) and session_id:
            return session_id
        return None

    def get_or_create_state(
        self, state_cls: type[BaseRealtimeState]
    ) -> BaseRealtimeState:
        if self.state is None:
            self.state = state_cls()
        elif not isinstance(self.state, state_cls):
            raise TypeError(
                f"Expected realtime state {state_cls.__name__}, "
                f"got {type(self.state).__name__}"
            )
        return self.state

    def dispose(self):
        if self.state is not None:
            self.state.dispose()
            self.state = None


class RealtimeSessionCache:
    def __init__(self, max_sessions: int = 64):
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, RealtimeSession] = OrderedDict()

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
        session_id = RealtimeSession.resolve_session_id(req)
        if session_id is None:
            return

        if req.block_idx == 0 or session_id not in self._sessions:
            if req.block_idx > 0:
                logger.warning(
                    "Missing realtime session state for session_id=%s (block_idx=%s). "
                    "Resetting block_idx to 0.",
                    session_id,
                    req.block_idx,
                )
                req.block_idx = 0
            self._sessions[session_id] = req.session or RealtimeSession()

        req.session = self._sessions[session_id]
        self._sessions.move_to_end(session_id)
        self._evict_stale_sessions()

    def _evict_stale_sessions(self) -> None:
        while len(self._sessions) > self.max_sessions:
            stale_session_id, stale_session = self._sessions.popitem(last=False)
            self._dispose_session(stale_session_id, stale_session)
            logger.debug("Evicted stale realtime session cache: %s", stale_session_id)
