"""Session radix cache (``--enable-session-radix-cache``): tag each request's KV
by session_id; ``release_session`` (close) frees a session's tagged KV.
# TODO (zhangmj): need to support priority_scheduling"""

from __future__ import annotations

import heapq
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Set

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

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
        self.unused_evictable_leaves: Set[TreeNode] = set()
        self.referenced_evictable_leaves: Set[TreeNode] = set()
        self.unused_evictable_size_: int = 0
        self.referenced_evictable_size_: int = 0
        self.session_id_to_ref_nodes: Dict[str, Set[TreeNode]] = {}
        self._closed_session_ids: OrderedDict = OrderedDict()

    def _tier_leaf_set(self, tier: int) -> set:
        if tier == TIER_UNUSED:
            return self.unused_evictable_leaves
        return self.referenced_evictable_leaves

    def _add_tier_size(self, tier: int, delta: int) -> None:
        if tier == TIER_UNUSED:
            self.unused_evictable_size_ += delta
        else:
            self.referenced_evictable_size_ += delta

    def _update_session_leaf_status(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        self.unused_evictable_leaves.discard(node)
        self.referenced_evictable_leaves.discard(node)
        if node.evicted or node.lock_ref > 0:
            return
        for child in node.children.values():
            if not child.evicted:
                return
        self._tier_leaf_set(_classify_node_tier(node)).add(node)

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
        if child.tracked_session_ids:
            new_node.tracked_session_ids = set(child.tracked_session_ids)
            for session_id in new_node.tracked_session_ids:
                nodes = self.session_id_to_ref_nodes.get(session_id)
                if nodes is not None:
                    nodes.add(new_node)
        self._update_session_leaf_status(new_node)
        self._update_session_leaf_status(child)

    def _session_forget_node(self, node: TreeNode) -> None:
        if not self.enable_session_radix_cache:
            return
        self.unused_evictable_leaves.discard(node)
        self.referenced_evictable_leaves.discard(node)
        if node.tracked_session_ids:
            for session_id in node.tracked_session_ids:
                nodes = self.session_id_to_ref_nodes.get(session_id)
                if nodes is not None:
                    nodes.discard(node)
            node.tracked_session_ids = None

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
        if not self.enable_session_radix_cache:
            return
        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        ref_nodes = self.session_id_to_ref_nodes.get(session_id)
        tracked_nodes = ref_nodes if ref_nodes is not None else set()

        last_node = getattr(req, "last_node", None)
        if last_node not in (None, self.root_node):
            new_nodes = self._collect_untracked_nodes_from_last_node(
                last_node, tracked_nodes
            )
        else:
            from sglang.srt.mem_cache.radix_cache import RadixKey

            token_ids = (req.origin_input_ids + req.output_ids)[: req.kv_committed_len]
            if not token_ids:
                return
            radix_key = RadixKey(
                token_ids, getattr(req, "extra_key", None)
            ).page_aligned(self.page_size)
            if len(radix_key) == 0:
                return
            nodes_on_path = self._collect_nodes_on_path(radix_key)
            new_nodes = [node for node in nodes_on_path if node not in tracked_nodes]

        if new_nodes:
            if ref_nodes is None:
                ref_nodes = self.session_id_to_ref_nodes.setdefault(session_id, set())
            for node in new_nodes:
                self._inc_session_ref(node)
                ref_nodes.add(node)
                if node.tracked_session_ids is None:
                    node.tracked_session_ids = set()
                node.tracked_session_ids.add(session_id)

    def _collect_nodes_on_path(self, key: RadixKey) -> list:
        node = self.root_node
        nodes = []
        while len(key) > 0:
            child_key = key.child_key(self.page_size)
            if child_key not in node.children:
                break
            child = node.children[child_key]
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len <= 0:
                break
            nodes.append(child)
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
        return nodes

    def _collect_untracked_nodes_from_last_node(
        self, node: Optional[TreeNode], tracked_nodes: Set[TreeNode]
    ) -> list:
        nodes = []
        while node not in (None, self.root_node):
            if node in tracked_nodes:
                break
            nodes.append(node)
            node = node.parent
        return nodes

    def _inc_session_ref(self, node: TreeNode) -> None:
        old_tier = _classify_node_tier(node)
        node.session_ref += 1
        self._maybe_move_node_tier(node, old_tier)

    def _dec_session_ref(self, node: TreeNode) -> None:
        old_tier = _classify_node_tier(node)
        node.session_ref = max(0, node.session_ref - 1)
        self._maybe_move_node_tier(node, old_tier)

    def _maybe_move_node_tier(self, node: TreeNode, old_tier: int) -> None:
        new_tier = _classify_node_tier(node)
        if new_tier == old_tier or node.evicted or node.lock_ref > 0:
            return
        node_size = len(node.key)
        old_set = self._tier_leaf_set(old_tier)
        if node in old_set:
            old_set.discard(node)
            if all(c.evicted for c in node.children.values()):
                self._tier_leaf_set(new_tier).add(node)
        self._add_tier_size(old_tier, -node_size)
        self._add_tier_size(new_tier, node_size)

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> None:
        if not self.enable_session_radix_cache:
            return
        self._closed_session_ids.pop(session_id, None)

    def release_radix_session(self, session_id: str) -> int:
        # Only release reference instead of evicting session's KV now
        if not self.enable_session_radix_cache or session_id is None:
            return 0
        self._remember_closed_session(session_id)
        ref_nodes = self.session_id_to_ref_nodes.pop(session_id, None)
        if not ref_nodes:
            return 0
        for node in ref_nodes:
            self._dec_session_ref(node)
            if node.tracked_session_ids is not None:
                node.tracked_session_ids.discard(session_id)
                if not node.tracked_session_ids:
                    node.tracked_session_ids = None
        logger.info(
            "release_radix_session %s: dereferenced %d nodes",
            session_id,
            len(ref_nodes),
        )
        return len(ref_nodes)

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

    # TODO (zhangmj): need to support priority_scheduling with three-tier eviction
    def _get_tier_priority(self, node: TreeNode, target_tier: int):
        # evict tagged KV with reference counter now
        return (node.session_ref, self.eviction_strategy.get_priority(node))

    def _evict_from_tier_heap(
        self, num_tokens: int, leaf_set: set, target_tier: int, evict_one_fn
    ) -> int:
        eviction_heap = [
            (self._get_tier_priority(node, target_tier), node) for node in leaf_set
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, x = heapq.heappop(eviction_heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue

            num_evicted += evict_one_fn(x)

            for child in x.parent.children.values():
                if not child.evicted:
                    break
            else:
                if x.parent.lock_ref == 0 and x.parent is not self.root_node:
                    if _classify_node_tier(x.parent) == target_tier:
                        heapq.heappush(
                            eviction_heap,
                            (self._get_tier_priority(x.parent, target_tier), x.parent),
                        )
        return num_evicted

    def _evict_one_device(self, node: TreeNode) -> int:
        self.token_to_kv_pool_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        self._record_remove_event(node)
        return num_evicted

    def _evict_tiered_device(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        num_evicted = self._evict_from_tier_heap(
            num_tokens,
            self.unused_evictable_leaves,
            TIER_UNUSED,
            self._evict_one_device,
        )
        if num_evicted < num_tokens:
            num_evicted += self._evict_from_tier_heap(
                num_tokens - num_evicted,
                self.referenced_evictable_leaves,
                TIER_REF,
                self._evict_one_device,
            )

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)


class SessionHiRadixCacheMixin(SessionRadixCacheMixin):
    """Session radix cache for HiRadixCache"""

    def _evict_tiered(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        num_evicted = self._evict_from_tier(
            num_tokens, self.unused_evictable_leaves, TIER_UNUSED
        )
        if num_evicted < num_tokens:
            num_evicted += self._evict_from_tier(
                num_tokens - num_evicted, self.referenced_evictable_leaves, TIER_REF
            )

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_from_tier(self, num_tokens: int, leaf_set: set, target_tier: int) -> int:
        if self.cache_controller.write_policy == "write_back":
            return self._evict_from_tier_write_back(num_tokens, leaf_set, target_tier)
        return self._evict_from_tier_write_through(num_tokens, leaf_set, target_tier)

    def _make_tier_eviction_heap(self, leaf_set: set, target_tier: int):
        heap = [(self._get_tier_priority(node, target_tier), node) for node in leaf_set]
        heapq.heapify(heap)
        return heap

    def _promote_tier_parent(self, node, heap, target_tier: int) -> None:
        p = node.parent
        if (
            p is not self.root_node
            and _classify_node_tier(p) == target_tier
            and all(c.evicted for c in p.children.values())
        ):
            heapq.heappush(heap, (self._get_tier_priority(p, target_tier), p))

    def _evict_from_tier_write_through(
        self, num_tokens: int, leaf_set: set, target_tier: int
    ) -> int:
        heap = self._make_tier_eviction_heap(leaf_set, target_tier)
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            else:
                num_evicted += self._evict_regular(x)
            self._promote_tier_parent(x, heap, target_tier)
        return num_evicted

    def _evict_from_tier_write_back(
        self, num_tokens: int, leaf_set: set, target_tier: int
    ) -> int:
        heap = self._make_tier_eviction_heap(leaf_set, target_tier)
        num_evicted = 0
        staged = []

        def flush_staged() -> None:
            if not staged:
                return
            self.writing_check(write_back=True)
            for node, device_indices in staged:
                self.cache_controller.evict_device(device_indices)
                node.release_host()
            staged.clear()

        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif self.write_backup(x, write_back=True) > 0:
                x.protect_host()
                staged.append((x, x.value))
                num_evicted += self._detach_backuped(x)
            else:
                flush_staged()
                num_evicted += self._drop_subtree_no_host(x)
            self._promote_tier_parent(x, heap, target_tier)
        flush_staged()
        return num_evicted

    def _evict_host_tiered(self, num_tokens: int) -> int:
        num_evicted = self._evict_host_from_tier(num_tokens, TIER_UNUSED)
        if num_evicted < num_tokens:
            num_evicted += self._evict_host_from_tier(
                num_tokens - num_evicted, TIER_REF
            )
        return num_evicted

    def _evict_host_from_tier(self, num_tokens: int, target_tier: int) -> int:
        from sglang.srt.disaggregation.kv_events import StorageMedium

        leaves = [
            n
            for n in self.evictable_host_leaves
            if n.evicted
            and n.host_ref_counter == 0
            and _classify_node_tier(n) == target_tier
        ]
        eviction_heap = [
            (self._get_tier_priority(node, target_tier), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            if not x.evicted or x.host_ref_counter > 0:
                continue
            if _classify_node_tier(x) != target_tier:
                continue

            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._session_forget_node(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                if _classify_node_tier(x.parent) == target_tier:
                    heapq.heappush(
                        eviction_heap,
                        (self._get_tier_priority(x.parent, target_tier), x.parent),
                    )

        return num_evicted
