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
    """Tags radix KV by session id; ``release_session`` (close) frees a session's
    tagged chains. A node holds the set of sessions on it, so a node shared by
    several sessions is freed only when its last holder closes. Tagged KV is
    ordinary LRU radix -- no pinning, no open. Mixed into RadixCache."""

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        """Add this request's session id to its leaf's holder set; no-op for non-session reqs."""
        sid = getattr(req, "session_id", None)
        if sid is None:
            return
        if node is None:
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
        if node is not None and node is not self.root_node:
            if node.session_ids is None:
                node.session_ids = set()
            node.session_ids.add(sid)
            logger.debug("tag session %s: node=%d holders=%d", sid, node.id, len(node.session_ids))

    def release_session(self, session_id: str) -> int:
        """Close: drop this session from each of its tagged leaves, freeing a node
        only once no other session still holds it (last holder). Shared
        prefixes/leaves kept. No index, so a late request can't leak. Synchronous."""
        pending = [
            n
            for n in self.evictable_leaves
            if n.session_ids is not None
            and session_id in n.session_ids
            and n.lock_ref == 0
        ]
        freed = 0
        while pending:
            node = pending[-1]
            if node.session_ids is not None:
                node.session_ids.discard(session_id)
            if (
                node is self.root_node
                or node.lock_ref != 0
                or len(node.children) != 0
                or node not in self.evictable_leaves
                or node.session_ids  # another session still holds this node
            ):
                pending.pop()  # chain done, hit a branch/lock/other holder, or evicted
                continue
            parent = node.parent
            self.token_to_kv_pool_allocator.free(node.value)
            self._delete_leaf(node)
            freed += 1
            pending[-1] = parent  # walk up the chain
        logger.info("release_session %s: freed %d nodes", session_id, freed)
        return freed
