"""Session radix cache (``--enable-session-radix-cache``): tag each request's KV
by session_id; ``release_session`` (close) frees a session's tagged KV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SessionRadixCacheMixin:
    """Tags radix KV by ``session_id``; ``release_session`` (close) frees a
    session's tagged chains. Tagged KV is ordinary LRU radix -- no pinning, no
    open. Mixed into RadixCache."""

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
        """Free a session's tagged KV on close: find its leaves by tag, free each
        unique leaf->branch chain (shared prefixes kept). No index, so a late
        request can't leak. Synchronous."""
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
