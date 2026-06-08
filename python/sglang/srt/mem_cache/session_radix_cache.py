"""Session radix cache (``--enable-session-radix-cache``): each request's KV is
tagged with its session_id. ``release_session`` (close) frees a session's tagged
KV; the tag is the primitive that more session-scoped ops can build on. Mixed
into RadixCache."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SessionRadixCacheMixin:
    """A session is a *tag* on radix KV (``node.session_id``), set as requests are
    cached -- not an eviction class. ``release_session`` (close) frees a session's
    tagged leaf->branch chains and is the first session-scoped op; more ops can
    build on the same tag. Tagged KV stays ordinary radix KV (LRU-neutral, no
    pinning), so it can't wedge the pool and needs no open/registration. Uses the
    host RadixCache's ``match_prefix``, ``evictable_leaves``,
    ``token_to_kv_pool_allocator``, ``_delete_leaf``, ``root_node``."""

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        """Tag this request's leaf with its session_id; no-op for non-session reqs."""
        sid = getattr(req, "session_id", None)
        if sid is None:
            return
        if node is None:
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
        if node is not None and node is not self.root_node:
            node.session_id = sid
            logger.debug("tag session %s: node=%d len=%d", sid, node.id, len(node.key))

    def release_session(self, session_id: str) -> int:
        """Free the session's tagged KV (the close op). Finds the session's leaves
        by scanning for the tag -- no index, so a request that finishes after close
        can't leak -- then frees each one's unique leaf->branch chain; shared
        prefixes are branch points and are preserved. Synchronous; a session is a
        handful of nodes."""
        pending = [
            n
            for n in self.evictable_leaves
            if getattr(n, "session_id", None) == session_id and n.lock_ref == 0
        ]
        freed = 0
        while pending:
            node = pending[-1]
            if (
                node is self.root_node
                or node.lock_ref != 0
                or len(node.children) != 0
                or node not in self.evictable_leaves
            ):
                pending.pop()  # chain done, hit a branch/lock, or already evicted
                continue
            parent = node.parent
            self.token_to_kv_pool_allocator.free(node.value)
            self._delete_leaf(node)
            freed += 1
            pending[-1] = parent  # walk up the chain
        logger.info("release_session %s: freed %d nodes", session_id, freed)
        return freed
