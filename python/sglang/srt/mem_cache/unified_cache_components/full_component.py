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
        self.cache.token_to_kv_pool_allocator.free(
            node.component_data[self.component_type].value
        )
        freed = len(node.component_data[self.component_type].value)
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
        cur = node
        while cur != self.cache.root_node:
            cd = cur.component_data[self.component_type]
            if cd.lock_ref == 0:
                self.cache.component_evictable_size_[self.component_type] -= len(
                    cd.value
                )
                self.cache.component_protected_size_[self.component_type] += len(
                    cd.value
                )
            cd.lock_ref += 1
            cur = cur.parent
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        cur = node
        while cur != self.cache.root_node:
            cd = cur.component_data[self.component_type]
            assert cd.lock_ref > 0
            if cd.lock_ref == 1:
                self.cache.component_evictable_size_[self.component_type] += len(
                    cd.value
                )
                self.cache.component_protected_size_[self.component_type] -= len(
                    cd.value
                )
            cd.lock_ref -= 1
            cur = cur.parent
