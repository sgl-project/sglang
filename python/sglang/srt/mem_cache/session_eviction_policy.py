"""Session-level eviction policy abstraction.

Policies implement ``score(session_id, metadata) -> float``; lower score means
evicted sooner. ``SessionMetadata`` carries the per-session state each policy
may inspect. New policies only need to subclass ``SessionEvictionPolicy`` and
override ``score`` — no changes to the eviction mechanism are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import msgspec


class SessionMetadata(msgspec.Struct):
    """Per-session state exposed to eviction policies."""

    last_active_time: float = 0.0
    priority: int = 0


class SessionEvictionPolicy(ABC):
    """Interface for session-level eviction ordering.

    ``score`` returns a float; sessions are evicted in ascending score order
    (lowest score first). Implementations may use any field on
    ``SessionMetadata`` or combine multiple signals.
    """

    @abstractmethod
    def score(self, session_id: str, metadata: SessionMetadata) -> float: ...


class LRUSessionEvictionPolicy(SessionEvictionPolicy):
    """Evict the least recently active session first."""

    def score(self, session_id: str, metadata: SessionMetadata) -> float:
        return metadata.last_active_time

# TODO(Zhangmj0621): Eviction based on priority.
# Pass priority into the session metadata and update it on each request.
class PriorityEvictionPolicy(SessionEvictionPolicy):
    pass
