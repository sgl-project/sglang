"""Session radix cache (``--enable-session-radix-cache``): tag each request's KV
by session_id; ``release_session`` (close) frees a session's tagged KV."""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Set

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit register_session clears the tombstone for intentional id reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192

# TODO (zhangmj): two tier strategy, need to support priority scheduling
TIER_UNUSED = 0
TIER_REF = 1


def _classify_node_tier(node: TreeNode) -> int:
    return TIER_REF if node.session_ref > 0 else TIER_UNUSED


class SessionRadixCacheMixin:
    """Tags radix KV by session id; ``release_session`` (close) releases a session's
    tagged reference. A node holds the set of sessions on it, so a node shared by
    several sessions is divided into unused nodes when its last holder closes. Tagged KV is
    evicted with reference counter -- no pinning, no open. Mixed into RadixCache."""

    def _reset_session_radix_state(self) -> None:
        self._session_leaves: Dict[str, Set[TreeNode]] = defaultdict(set)
        self._closed_session_ids: OrderedDict = OrderedDict()
        self._session_incarnation_counter: int = 0
        self._session_generations: Dict[str, int] = {}
        self.unused_evictable_size_: int = 0
        self.referenced_evictable_size_: int = 0

    def _ensure_session_radix_state(self) -> None:
        if not hasattr(self, "_session_leaves"):
            self._reset_session_radix_state()

    def _add_tier_size(self, tier: int, delta: int) -> None:
        if tier == TIER_UNUSED:
            self.unused_evictable_size_ += delta
        else:
            self.referenced_evictable_size_ += delta

    def _session_on_lock(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache or node.evicted:
            return
        self._add_tier_size(_classify_node_tier(node), -len(node.key))

    def _session_on_unlock(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache or node.evicted:
            return
        self._add_tier_size(_classify_node_tier(node), len(node.key))

    def _account_new_evictable_node(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        if node is None or node is self.root_node:
            return
        if node.evicted or node.lock_ref > 0:
            return
        self._add_tier_size(_classify_node_tier(node), len(node.key))

    def _session_on_split(self, new_node: TreeNode, child: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        new_node.session_ref = child.session_ref

    def _session_forget_node(self, node: TreeNode) -> None:
        session_ids = getattr(node, "session_ids", None)
        if not session_ids or not hasattr(self, "_session_leaves"):
            return
        parent = node.parent
        recede = parent is not None and parent is not self.root_node
        for sid in tuple(session_ids):
            leaves = self._session_leaves.get(sid)
            if leaves is None:
                continue
            leaves.discard(node)
            if recede:
                leaves.add(parent)
                parent_ids = getattr(parent, "session_ids", None)
                if parent_ids is None:
                    parent.session_ids = parent_ids = set()
                parent_ids.add(sid)
        del node.session_ids

    def _session_on_delete_leaf(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        if not node.evicted and node.lock_ref == 0:
            self._add_tier_size(_classify_node_tier(node), -len(node.key))
        self._session_forget_node(node)

    def _session_on_detach_backuped(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        if not node.evicted and node.lock_ref == 0:
            self._add_tier_size(_classify_node_tier(node), -len(node.key))

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = getattr(req, "session_id", None)
        if session_id is None:
            session_id = getattr(getattr(req, "session", None), "session_id", None)
        return session_id

    def register_session_ref(self, req: Req) -> None:
        """Add this request's session id to its leaf's holder set; no-op for non-session reqs."""
        if not self.enable_session_radix_cache:
            return
        self._ensure_session_radix_state()
        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        if (
            req.session_generation is not None
            and req.session_generation != self._session_generations.get(session_id)
        ):
            logger.warning("register_session_ref called for stale request; Skip it.")
            return

        node = getattr(req, "last_node", None)
        if node is None:
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
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node

        if node in (None, self.root_node):
            return
        if node in self._session_leaves[session_id]:
            return
        self._session_leaves[session_id].add(node)
        node_ids = getattr(node, "session_ids", None)
        if node_ids is None:
            node.session_ids = node_ids = set()
        node_ids.add(session_id)
        while node not in (None, self.root_node):
            old_tier = _classify_node_tier(node)
            node.session_ref += 1
            self._maybe_move_node_tier(node, old_tier)
            parent = node.parent
            if parent in self._session_leaves[session_id]:
                self._session_leaves[session_id].discard(parent)
                parent_ids = getattr(parent, "session_ids", None)
                if parent_ids is not None:
                    parent_ids.discard(session_id)
                    if not parent_ids:
                        del parent.session_ids
                break
            node = parent

    def _maybe_move_node_tier(self, node: TreeNode, old_tier: int) -> None:
        new_tier = _classify_node_tier(node)
        if new_tier == old_tier or node.evicted or node.lock_ref > 0:
            return
        node_size = len(node.key)
        self._add_tier_size(old_tier, -node_size)
        self._add_tier_size(new_tier, node_size)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        self._closed_session_ids.pop(session_id, None)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        return self._session_incarnation_counter

    def current_session_generation(self, session_id: str) -> Optional[int]:
        return self._session_generations.get(session_id)

    def release_radix_session(self, session_id: str) -> int:
        # TODO(zhangmj): distinguish between agents and directly free kv for subagent
        if not self.enable_session_radix_cache or session_id is None:
            return 0
        self._ensure_session_radix_state()
        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)
        indexed = self._session_leaves.pop(session_id, set())
        freed = 0
        for leaf in indexed:
            if session_id not in getattr(leaf, "session_ids", set()):
                continue
            node = leaf
            while node not in (None, self.root_node):
                session_ids = getattr(node, "session_ids", None)
                if session_ids is not None:
                    session_ids.discard(session_id)
                    if not session_ids:
                        delattr(node, "session_ids")
                old_tier = _classify_node_tier(node)
                node.session_ref -= 1
                self._maybe_move_node_tier(node, old_tier)
                node = node.parent
        logger.info(
            "release_session %s: indexed %d leaves, freed %d nodes",
            session_id,
            len(indexed),
            freed,
        )
        # Note: only return 0 temporarily since we do not evict KV when release session now
        return freed

    def available_and_evictable_str(self) -> str:
        if not self.enable_session_radix_cache:
            return super().available_and_evictable_str()
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        protected_size = self.protected_size()
        return (
            f"Available tokens: {available_size + evictable_size} "
            f"({available_size=} + {evictable_size=}, "
            f"unused_evictable_size={self.unused_evictable_size_}, "
            f"referenced_evictable_size={self.referenced_evictable_size_}, "
            f"{protected_size=})\n"
        )
