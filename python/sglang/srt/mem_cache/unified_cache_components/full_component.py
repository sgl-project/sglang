from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        super().__init__(cache, params)
        allocator = cache.token_to_kv_pool_allocator
        # When SWA is present, only free full-attention KV here;
        # SWA KV will be freed by cascade via SWAComponent.evict_component.
        if ComponentType.SWA in cache.tree_components:
            self._free_full = allocator.full_attn_allocator.free
        else:
            self._free_full = allocator.free

    def node_has_component_data(self, node: UnifiedTreeNode) -> bool:
        # Override so _for_each_component_lru includes Full in LRU operations
        return node.component_data[self.component_type].value is not None

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        return lambda node: True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].lock_ref = child.component_data[
            self.component_type
        ].lock_ref

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        cd = node.component_data[self.component_type]
        self._free_full(cd.value)
        freed = len(cd.value)
        self.cache.component_evictable_size_[self.component_type] -= freed
        return freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.num_tokens
        lru = self.cache.lru_lists[self.component_type]
        while tracker[self.component_type] < request:
            x = lru.get_leaf_lru_no_lock()
            if x is None:
                break
            self.cache._evict_component_and_detach_lru(
                x, self, is_leaf=True, tracker=tracker
            )
            self.cache._cascade_evict(x, self, tracker)

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        root = self.cache.root_node
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] -= key_len
                self.cache.component_protected_size_[ct] += key_len
            cd.lock_ref += 1
            cur = cur.parent
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        root = self.cache.root_node
        cur = node
        while cur != root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0
            if cd.lock_ref == 1:
                key_len = len(cd.value)
                self.cache.component_evictable_size_[ct] += key_len
                self.cache.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            cur = cur.parent
