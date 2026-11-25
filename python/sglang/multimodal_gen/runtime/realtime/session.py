# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def resident_tensor_bytes(roots) -> int:
    storages = {}
    visited = set()

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type not in ("cpu", "meta"):
                storage = value.untyped_storage()
                storages[(value.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif is_dataclass(value) and not isinstance(value, type):
            for name in fields(value):
                visit(getattr(value, name.name))

    visit(roots)
    return sum(storages.values())


class BaseRealtimeState:
    """per-session state owned by pipeline stages"""

    def tensor_roots(self):
        return vars(self)

    def reserved_bytes(self) -> int:
        return 0

    def fork(self):
        """Action policies replace retained tensors and fork mutable containers."""
        return copy(self)

    def committed(self) -> None:
        """Publish work that must only start once the action response is ready."""

    def discard(self, previous: BaseRealtimeState | None) -> None:
        if previous is None:
            self.dispose()

    def dispose(self) -> None:
        pass


class RealtimeSession:
    """reusable state container across realtime request chunks"""

    def __init__(self) -> None:
        self.last_action_step: int | None = None
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

    def fork(self):
        session = RealtimeSession()
        session.last_action_step = self.last_action_step
        session._states = {kind: state.fork() for kind, state in self._states.items()}
        return session

    def resident_bytes(self):
        roots = [state.tensor_roots() for state in self._states.values()]
        return resident_tensor_bytes(roots) + sum(
            state.reserved_bytes() for state in self._states.values()
        )

    def committed(self):
        for state in self._states.values():
            state.committed()

    def discard(self, previous):
        for kind, state in self._states.items():
            state.discard(previous.get_state(kind) if previous is not None else None)

    def dispose(self) -> None:
        for state in list(self._states.values()):
            state.dispose()
        self._states.clear()


class RealtimeSessionCache:
    """lru cache that binds incoming chunks to persistent realtime sessions"""

    def __init__(self, max_sessions: int = 64, max_bytes: int | None = None) -> None:
        if max_sessions < 1 or (max_bytes is not None and max_bytes <= 0):
            raise ValueError("Session limits must be positive")
        self.max_bytes = max_bytes
        self._sizes = {}
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
        self._sizes.pop(session_id, None)
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

        if getattr(req, "extra", {}).get("action_session"):
            old = self._sessions.get(session_id)
            step = req.extra.get("action_realtime", {}).get("start_step")
            if (
                step is not None
                and old is not None
                and old.last_action_step is not None
                and step <= old.last_action_step
            ):
                raise ValueError(
                    "action_chunk_conflict: a chunk for this or a later control step already committed"
                )
            req.session = old.fork() if old is not None else RealtimeSession()
            if step is not None:
                req.session.last_action_step = step
            return

        if session_id not in self._sessions:
            if req.block_idx > 0:
                raise ValueError(
                    "Missing realtime session state for "
                    f"session_id={session_id} block_idx={req.block_idx}."
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
        self._sessions.move_to_end(session_id)
        self._evict_stale_sessions()

    def discard(self, req):
        if not req.extra.get("action_session") or req.session is None:
            return
        previous = self._sessions.get(RealtimeSession.resolve_session_id(req))
        if previous is not req.session:
            req.session.discard(previous)
            req.session = None

    def commit(self, req):
        self.commit_many([req])

    def commit_many(self, requests):
        # Size every candidate before changing any committed session. A rejected
        # member of a tensor batch must leave its peers retryable.
        entries = []
        for req in requests:
            session_id = RealtimeSession.resolve_session_id(req)
            if session_id is None or req.session is None:
                continue
            size = req.session.resident_bytes() if self.max_bytes is not None else 0
            if self.max_bytes is not None and size > self.max_bytes:
                if not req.extra.get("action_session"):
                    self.release(session_id)
                raise ValueError("Session exceeds the retained GPU memory budget")
            entries.append((session_id, req.session, size))
        for session_id, session, size in entries:
            self._sessions[session_id] = session
            self._sizes[session_id] = size
            self._sessions.move_to_end(session_id)
            session.committed()
        self._evict_stale_sessions()

    def _evict_stale_sessions(self) -> None:
        while len(self._sessions) > self.max_sessions or (
            self.max_bytes is not None and sum(self._sizes.values()) > self.max_bytes
        ):
            stale_session_id, stale_session = self._sessions.popitem(last=False)
            self._sizes.pop(stale_session_id, None)
            self._dispose_session(stale_session_id, stale_session)
            logger.debug("Evicted stale realtime session cache: %s", stale_session_id)
