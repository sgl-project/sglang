"""Session radix cache (``--enable-session-radix-cache``): tag each request's KV
by session_id; ``release_session`` (close) frees a session's tagged KV;
``preempt_sessions`` evicts idle sessions under memory pressure via a pluggable
``SessionEvictionPolicy``."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.session_eviction_policy import (
    LRUSessionEvictionPolicy,
    SessionEvictionPolicy,
    SessionMetadata,
)

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

    def _reset_session_radix_state(
        self, eviction_policy: SessionEvictionPolicy | None = None
    ) -> None:
        self._session_leaves: defaultdict[str, set] = defaultdict(set)
        self._closed_session_ids: OrderedDict = OrderedDict()
        self._session_metadata: dict[str, SessionMetadata] = {}
        self._session_eviction_policy: SessionEvictionPolicy = (
            eviction_policy or LRUSessionEvictionPolicy()
        )

    def _ensure_session_radix_state(self) -> None:
        if not hasattr(self, "_session_leaves"):
            self._reset_session_radix_state()

    def set_session_eviction_policy(self, policy: SessionEvictionPolicy) -> None:
        self._ensure_session_radix_state()
        self._session_eviction_policy = policy

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def register_session(self, session_id: str, priority: int = 0) -> None:
        self._ensure_session_radix_state()
        if session_id is None:
            return
        self._closed_session_ids.pop(session_id, None)
        self._session_leaves.setdefault(session_id, set())
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = SessionMetadata(priority=priority)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    # ------------------------------------------------------------------
    # Leaf tagging
    # ------------------------------------------------------------------

    def _discard_session_leaf(self, node) -> None:
        session_ids = getattr(node, "session_ids", None)
        if not session_ids:
            return
        if not hasattr(self, "_session_leaves"):
            logger.error(
                "_discard_session_leaf: node has session_ids but mixin state is "
                "uninitialised — session index is inconsistent"
            )
            return
        for sid in tuple(session_ids):
            leaves = self._session_leaves.get(sid)
            if leaves is None:
                logger.warning(
                    "_discard_session_leaf: node tagged with session %s but that "
                    "session is not in _session_leaves — index inconsistency",
                    sid,
                )
                continue
            leaves.discard(node)
            if not leaves and sid not in self._closed_session_ids:
                self._session_leaves.pop(sid, None)
        if hasattr(node, "session_ids"):
            delattr(node, "session_ids")

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        """Add this request's session id to its leaf's holder set; no-op for non-session reqs."""
        self._ensure_session_radix_state()
        sid = getattr(req, "session_id", None)
        if sid is None or sid in self._closed_session_ids:
            return
        if node is None:
            logger.warning(
                "_tag_session_leaf called without node; falling back to match_prefix"
            )
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
            if node is None or node is self.root_node:
                logger.error(
                    "_tag_session_leaf: could not resolve a node for session %s — "
                    "this turn's KV will not be tracked and is invisible to preemption",
                    sid,
                )
                return
        if node is not self.root_node:
            session_ids = getattr(node, "session_ids", None)
            if session_ids is None:
                session_ids = set()
                node.session_ids = session_ids
            session_ids.add(sid)
            self._session_leaves[sid].add(node)
            meta = self._session_metadata.setdefault(sid, SessionMetadata())
            meta.last_active_time = time.monotonic()
            logger.debug(
                "tag session %s: node=%d holders=%d indexed=%d",
                sid,
                node.id,
                len(session_ids),
                len(self._session_leaves[sid]),
            )

    # ------------------------------------------------------------------
    # Explicit close
    # ------------------------------------------------------------------

    def release_session(self, session_id: str) -> int:
        """Close: drop this session from each of its tagged leaves, freeing a node
        only once no other session still holds it (last holder). Shared
        prefixes/leaves kept."""
        self._ensure_session_radix_state()
        self._remember_closed_session(session_id)
        self._session_metadata.pop(session_id, None)
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

    # ------------------------------------------------------------------
    # Memory-pressure preemption
    # ------------------------------------------------------------------

    def preempt_session_kv(self, session_id: str) -> int:
        """Eviction mechanism: free a session's unique KV nodes without closing it.
        The session stays open; its next turn re-prefills from whatever the tree
        still has and re-tags from scratch. Returns freed token count."""
        self._ensure_session_radix_state()
        if session_id not in self._session_leaves:
            logger.warning(
                "preempt_session_kv: session %s not in _session_leaves; nothing to preempt",
                session_id,
            )
            return 0
        indexed = self._session_leaves.get(session_id, set())
        self._session_leaves[session_id] = set()
        freed_tokens = 0
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
                freed_tokens += len(node.value)
                self.token_to_kv_pool_allocator.free(node.value)
                self._delete_leaf(node)
                node = parent
        logger.info(
            "preempt_session_kv %s: freed %d tokens",
            session_id,
            freed_tokens,
        )
        return freed_tokens

    def preempt_sessions(self, num_tokens: int) -> int:
        """Eviction policy: score and rank idle sessions via the configured
        SessionEvictionPolicy, then call preempt_session_kv on victims until
        num_tokens are freed or all candidates are exhausted."""
        self._ensure_session_radix_state()

        def _get_metadata(sid: str) -> SessionMetadata:
            meta = self._session_metadata.get(sid)
            if meta is None:
                logger.error(
                    "preempt_sessions: session %s has leaves but no metadata — "
                    "treating as lowest priority to avoid silent wrong ordering",
                    sid,
                )
                return SessionMetadata()
            return meta

        candidates = sorted(
            (sid for sid, leaves in self._session_leaves.items() if leaves),
            key=lambda sid: self._session_eviction_policy.score(sid, _get_metadata(sid)),
        )
        freed = 0
        for sid in candidates:
            if freed >= num_tokens:
                break
            freed += self.preempt_session_kv(sid)
        logger.info(
            "preempt_sessions: target=%d freed=%d candidates=%d policy=%s",
            num_tokens,
            freed,
            len(candidates),
            type(self._session_eviction_policy).__name__,
        )
        return freed
