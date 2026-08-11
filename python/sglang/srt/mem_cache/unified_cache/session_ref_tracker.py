"""Session ref tracking for UnifiedRadixCache (``--enable-session-radix-cache``):
tag each request's KV by session_id for each tree component; ``release_radix_session``
(close) releases a session's tagged reference.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        TreeComponent,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit register_session clears the tombstone for intentional id reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192

SessionCachePriorityStatus = Literal[
    "updated", "unchanged", "not_found", "stale_generation", "disabled"
]


@dataclass(frozen=True, kw_only=True)
class SessionCachePriorityResult:
    status: SessionCachePriorityStatus
    generation: Optional[int]
    indexed_component_leaves: int = 0


@dataclass(kw_only=True)
class UnifiedSessionRefTracker:
    """Tags radix KV by session id; ``release_radix_session`` (close) releases a
    session's tagged reference. Each component maintains its own session_ids,
    session_ref and so on."""

    components: tuple[TreeComponent, ...]
    tree_core: UnifiedTreeCore
    enable_session_radix_cache: bool

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._closed_session_ids: OrderedDict[str, None] = OrderedDict()
        self._session_incarnation_counter: int = 0
        self._session_generations: dict[str, int] = {}
        self._demoted_session_ids: set[str] = set()
        for component in self.components:
            component.reset_session_state()

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = req.session_id
        if session_id is None and req.session is not None:
            session_id = req.session.session_id
        return session_id

    def register_session_ref(self, req: Req) -> None:
        """Register a non-streaming request's reusable leaves with each component."""
        if not self.enable_session_radix_cache:
            return

        session = req.session
        if session is not None and session.streaming:
            return

        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        current_generation = self._session_generations.get(session_id)
        if current_generation is None or req.session_generation != current_generation:
            logger.warning("register_session_ref called for stale request; Skip it.")
            return

        assert req.last_node is not None
        last_node = self.tree_core.node_by_id(req.last_node)
        if last_node is self.tree_core.root_node:
            return

        for component in self.components:
            leaf = component.resolve_session_leaf(req, last_node)
            component.register_session_leaf(session_id, leaf)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        self._closed_session_ids.pop(session_id, None)
        self._demoted_session_ids.discard(session_id)
        for component in self.components:
            component.set_session_protected(session_id, True)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        return self._session_incarnation_counter

    def ensure_session_generation(self, session_id: str) -> int:
        generation = self._session_generations.get(session_id)
        if generation is None:
            generation = self.open_radix_session(session_id)
        return generation

    def set_session_cache_priority(
        self,
        session_id: str,
        *,
        protected: bool,
        generation: Optional[int] = None,
    ) -> SessionCachePriorityResult:
        if not self.enable_session_radix_cache:
            return SessionCachePriorityResult(status="disabled", generation=None)

        current_generation = self._session_generations.get(session_id)
        if current_generation is None or session_id in self._closed_session_ids:
            return SessionCachePriorityResult(status="not_found", generation=None)
        if generation is not None and generation != current_generation:
            return SessionCachePriorityResult(
                status="stale_generation", generation=current_generation
            )

        was_protected = session_id not in self._demoted_session_ids
        indexed = 0
        for component in self.components:
            changed, component_leaves = component.set_session_protected(
                session_id, protected
            )
            assert changed == (protected != was_protected)
            indexed += component_leaves

        if protected:
            self._demoted_session_ids.discard(session_id)
        else:
            self._demoted_session_ids.add(session_id)

        return SessionCachePriorityResult(
            status="updated" if protected != was_protected else "unchanged",
            generation=current_generation,
            indexed_component_leaves=indexed,
        )

    def release_radix_session(self, session_id: str) -> int:
        if not self.enable_session_radix_cache or session_id is None:
            return 0

        if session_id in self._closed_session_ids:
            return 0

        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)
        self._demoted_session_ids.discard(session_id)

        indexed = 0
        for component in self.components:
            indexed += component.release_session(session_id)

        logger.info(
            "release_session %s: indexed %d component leaves",
            session_id,
            indexed,
        )
        return 0
