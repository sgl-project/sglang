"""Cache-level session lifecycle for the unified radix cache."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192


class SessionUnifiedRadixCacheMixin:
    """Coordinate session lifecycle across unified cache components."""

    def _reset_session_radix_state(self) -> None:
        self._closed_session_ids: OrderedDict[str, None] = OrderedDict()
        self._session_incarnation_counter: int = 0
        self._session_generations: dict[str, int] = {}
        for component in self._components_tuple:
            component.reset_session_state()

    def _ensure_session_radix_state(self) -> None:
        if not hasattr(self, "_closed_session_ids"):
            self._reset_session_radix_state()

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = getattr(req, "session_id", None)
        if session_id is None:
            session_id = getattr(getattr(req, "session", None), "session_id", None)
        return session_id

    def register_session_ref(self, req: Req) -> None:
        """Register a non-streaming request's reusable leaves with each component."""
        if not self.enable_session_radix_cache:
            return

        session = getattr(req, "session", None)
        if session is not None and getattr(session, "streaming", False):
            return

        self._ensure_session_radix_state()
        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        if getattr(
            req, "session_generation", None
        ) is not None and req.session_generation != self._session_generations.get(
            session_id
        ):
            logger.warning("register_session_ref called for stale request; Skip it.")
            return

        last_node = getattr(req, "last_node", None)
        if last_node is None:
            logger.warning(
                "register_session_ref called for request without last_node; falling back to match_prefix."
            )
            from sglang.srt.mem_cache.radix_cache import RadixKey

            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return
            radix_key = RadixKey(
                token_ids, getattr(req, "extra_key", None)
            ).page_aligned(self.page_size)
            if len(radix_key) == 0:
                return
            last_node = self.match_prefix(
                MatchPrefixParams(key=radix_key)
            ).last_device_node

        if last_node in (None, self.root_node):
            return

        for component in self._components_tuple:
            leaf = component.resolve_session_leaf(req, last_node)
            component.register_session_leaf(session_id, leaf)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        self._ensure_session_radix_state()
        self._closed_session_ids.pop(session_id, None)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        return self._session_incarnation_counter

    def current_session_generation(self, session_id: str) -> Optional[int]:
        self._ensure_session_radix_state()
        return self._session_generations.get(session_id)

    def release_radix_session(self, session_id: str) -> int:
        if not self.enable_session_radix_cache or session_id is None:
            return 0

        self._ensure_session_radix_state()
        if session_id in self._closed_session_ids:
            return 0

        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)

        indexed = 0
        for component in self._components_tuple:
            indexed += component.release_session(session_id)

        logger.info(
            "release_session %s: indexed %d component leaves",
            session_id,
            indexed,
        )
        return 0
