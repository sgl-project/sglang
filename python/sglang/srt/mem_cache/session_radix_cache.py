"""Session radix cache (``--enable-session-radix-cache``): tag each request's KV
by session_id; ``release_session`` (close) frees a session's tagged KV."""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit register_session clears the tombstone for intentional id reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192


class SessionRadixCacheMixin:
    """Tags radix KV by session id; ``release_session`` (close) frees a session's
    tagged chains. A node holds the set of sessions on it, so a node shared by
    several sessions is freed only when its last holder closes. Tagged KV is
    ordinary LRU radix -- no pinning, no open. Mixed into RadixCache."""

    def _reset_session_radix_state(self) -> None:
        self._session_leaves = defaultdict(set)
        self._closed_session_ids = OrderedDict()

    def _ensure_session_radix_state(self) -> None:
        if not hasattr(self, "_session_leaves"):
            self._reset_session_radix_state()

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def _discard_session_leaf(self, node) -> None:
        session_ids = getattr(node, "session_ids", None)
        if not session_ids or not hasattr(self, "_session_leaves"):
            return
        for sid in tuple(session_ids):
            leaves = self._session_leaves.get(sid)
            if leaves is not None:
                leaves.discard(node)
                if not leaves and sid not in self._closed_session_ids:
                    self._session_leaves.pop(sid, None)
        if hasattr(node, "session_ids"):
            delattr(node, "session_ids")

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        """Add this request's session id to its leaf's holder set; no-op for non-session reqs."""
        if not self.enable_session_radix_cache:
            return
        self._ensure_session_radix_state()
        sid = req.session_id
        if sid is None or sid in self._closed_session_ids:
            return
        if node is None:
            logger.warning(
                "_tag_session_leaf called without node; falling back to match_prefix"
            )
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
        if node is not None and node is not self.root_node:
            session_ids = getattr(node, "session_ids", None)
            if session_ids is None:
                session_ids = set()
                node.session_ids = session_ids
            session_ids.add(sid)
            self._session_leaves[sid].add(node)
            logger.debug(
                "tag session %s: node=%d holders=%d indexed=%d",
                sid,
                node.id,
                len(session_ids),
                len(self._session_leaves[sid]),
            )

    def release_radix_session(self, session_id: str) -> int:
        """Close: drop this session from each of its tagged leaves, freeing a node
        only once no other session still holds it (last holder). Shared
        prefixes/leaves kept."""
        self._ensure_session_radix_state()
        self._remember_closed_session(session_id)
        indexed = self._session_leaves.pop(session_id, set())
        freed = 0
        for leaf in indexed:
            if session_id not in getattr(leaf, "session_ids", set()):
                continue
            node = leaf
            while True:
                session_ids = getattr(node, "session_ids", None)
                if session_ids is not None:
                    session_ids.discard(session_id)
                    if not session_ids:
                        delattr(node, "session_ids")
                if (
                    node is self.root_node
                    or node.lock_ref != 0
                    or len(node.children) != 0
                    or node not in self.evictable_leaves
                    or getattr(node, "session_ids", None)
                ):
                    break
                parent = node.parent
                self.token_to_kv_pool_allocator.free(node.value)
                self._delete_leaf(node)
                freed += 1
                node = parent
        logger.info(
            "release_session %s: indexed %d leaves, freed %d nodes",
            session_id,
            len(indexed),
            freed,
        )
        return freed
