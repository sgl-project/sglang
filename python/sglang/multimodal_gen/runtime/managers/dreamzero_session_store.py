# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class SessionState:
    """Worker-local state retained across DreamZero observations."""

    kv_cache1: list[torch.Tensor] = field(default_factory=list)
    kv_cache_neg: list[torch.Tensor] = field(default_factory=list)
    crossattn_cache: list[dict[str, Any]] = field(default_factory=list)
    crossattn_cache_neg: list[dict[str, Any]] = field(default_factory=list)

    clip_feas: torch.Tensor | None = None
    ys: torch.Tensor | None = None
    latent_video: torch.Tensor | None = None

    current_start_frame: int = 0
    language: torch.Tensor | None = None
    negative_language: torch.Tensor | None = None
    local_attn_size: int = -1
    cached_prompt_embs: list[torch.Tensor] | None = None

    def reset_stream(self, *, preserve_text: bool) -> None:
        """Start a new sequence while optionally retaining encoded text."""

        self.kv_cache1.clear()
        self.kv_cache_neg.clear()
        self.crossattn_cache.clear()
        self.crossattn_cache_neg.clear()
        self.clip_feas = None
        self.ys = None
        self.latent_video = None
        self.current_start_frame = 0
        if not preserve_text:
            self.language = None
            self.negative_language = None
            self.cached_prompt_embs = None


class SessionStore:
    """Bounded LRU store for DreamZero GPU-resident session state."""

    def __init__(self, max_sessions: int = 10) -> None:
        if max_sessions < 1:
            raise ValueError("DreamZero max_sessions must be at least 1")
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    def get_or_create(
        self,
        session_id: str,
        factory: Callable[[], SessionState],
    ) -> SessionState:
        if not session_id:
            raise ValueError("DreamZero session_id must be a non-empty string")
        state = self._sessions.get(session_id)
        if state is not None:
            self._sessions.move_to_end(session_id)
            return state

        if len(self._sessions) >= self.max_sessions:
            self.evict_lru()
        state = factory()
        self._sessions[session_id] = state
        return state

    def reset(self, session_id: str) -> SessionState | None:
        return self._sessions.pop(session_id, None)

    def evict_lru(self) -> tuple[str, SessionState] | None:
        if not self._sessions:
            return None
        session_id, state = self._sessions.popitem(last=False)
        logger.warning(
            "Evicting DreamZero session %r from worker-local LRU cache",
            session_id,
        )
        return session_id, state

    def get_active_count(self) -> int:
        return len(self._sessions)


def get_request_session_state(
    batch,
    session_store: SessionStore | None,
    *,
    local_attn_size: int,
) -> SessionState:
    """Resolve persistent or request-local state and attach it to ``batch``."""

    existing = getattr(batch, "dreamzero_session_state", None)
    if isinstance(existing, SessionState):
        return existing

    session_id = getattr(batch, "dreamzero_session_id", None)
    if session_id is None:
        session_id = getattr(batch, "session_id", None)
    sampling_params = getattr(batch, "sampling_params", None)
    if session_id is None and sampling_params is not None:
        session_id = getattr(sampling_params, "session_id", None)
    if session_id is not None:
        session_id = str(session_id).strip() or None

    def factory() -> SessionState:
        return SessionState(local_attn_size=local_attn_size)

    if session_id is None or session_store is None:
        state = factory()
        persistent = False
    else:
        reset_already_applied = bool(
            getattr(batch, "dreamzero_session_reset_applied", False)
        )
        reset_session = bool(getattr(batch, "reset_session", False))
        if sampling_params is not None:
            reset_session = reset_session or bool(
                getattr(sampling_params, "reset_session", False)
            )
        reset_session = reset_session or bool(
            getattr(batch, "extra", {}).get("dreamzero_reset_session", False)
        )
        if reset_session and not reset_already_applied:
            session_store.reset(session_id)
        batch.dreamzero_session_reset_applied = (
            reset_already_applied or reset_session
        )
        state = session_store.get_or_create(session_id, factory)
        persistent = True

    batch.dreamzero_session_id = session_id
    batch.dreamzero_session_state = state
    batch.dreamzero_session_persistent = persistent
    return state
