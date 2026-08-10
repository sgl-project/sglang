"""C128 sidecar ownership for the DSV4 NPU Unified Radix Cache.

One sidecar page contains the 16 C128 KV entries produced by a 2048-token
Full prefix group.  The component deliberately exposes only complete groups to
the radix tree: partial tail pages remain request-owned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedTreeNode,
    )


C128_GROUP_TOKENS = 2048


class C128SidecarComponent(TreeComponent):
    component_type = ComponentType.C128

    @property
    def allocator(self):
        return self.cache.token_to_kv_pool_allocator

    def _attach(self, node: UnifiedTreeNode, pages: torch.Tensor) -> None:
        if pages.numel() == 0:
            return
        ct = self.component_type
        cd = node.component_data[ct]
        assert cd.value is None
        value = pages.clone()
        self.tree_core.set_component_device_value(node.id, ct, value)
        self.allocator.retain_c128_pages(value)

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        # A page is attached only to the node ending its 2048-token group.
        return lambda node: node.component_data[self.component_type].value is not None

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        req = params.req
        if req is None:
            return result

        chunks = []
        node = self.tree_core.node_by_id(result.best_match_node)
        root = self.tree_core.root_node
        while node is not root:
            value = node.component_data[self.component_type].value
            if value is not None:
                chunks.append(value)
            node = node.parent
        chunks.reverse()
        pages = (
            torch.cat(chunks)
            if chunks
            else self.allocator.c128_attn_allocator.free_pages.new_empty((0,))
        )
        assert pages.numel() == len(result.device_indices) // C128_GROUP_TOKENS
        self.cache.req_to_token_pool.set_c128_prefix_pages(req, pages)
        return result

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        pages = params.c128_value
        assert pages is not None
        start = total_prefix_len // C128_GROUP_TOKENS
        end = (total_prefix_len + prefix_len) // C128_GROUP_TOKENS
        self._attach(node, pages[start:end])

    @staticmethod
    def _node_depth(node: UnifiedTreeNode) -> int:
        depth = 0
        while node.parent is not None:
            depth += len(node.key)
            node = node.parent
        return depth

    @staticmethod
    def _split_pending_swa_rebuild(
        new_parent: UnifiedTreeNode,
        child: UnifiedTreeNode,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Keep a deferred SWA rebuild aligned when C128 splits its source node."""
        for i, pending in enumerate(cache_actions):
            if isinstance(pending, SWARebuild) and pending.node_id == child.id:
                cache_actions[i : i + 1] = [
                    SWARebuild(
                        new_parent.id,
                        new_parent.component_data[BASE_COMPONENT_TYPE].value,
                    ),
                    SWARebuild(
                        child.id,
                        child.component_data[BASE_COMPONENT_TYPE].value,
                    ),
                ]
                return

    def _ensure_boundary_node(
        self,
        tail: UnifiedTreeNode,
        boundary: int,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> UnifiedTreeNode:
        node = tail
        while node.parent is not None and self._node_depth(node.parent) >= boundary:
            node = node.parent

        node_end = self._node_depth(node)
        if node_end == boundary:
            return node

        node_start = node_end - len(node.key)
        assert node_start < boundary < node_end
        new_parent, action = self.tree_core._split_node(
            node.key, node, boundary - node_start
        )
        if action is not None:
            cache_actions.append(action)
        self._split_pending_swa_rebuild(new_parent, node, cache_actions)
        return new_parent

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        if not is_new_leaf:
            return
        assert params.key is not None
        assert params.c128_value is not None

        # Full/SWA may initially represent the new suffix as one long leaf.
        # Materialize every complete C128 group boundary so a later branch can
        # always match the nearest 2048-token prefix instead of falling back to
        # the previous, potentially much shorter, Radix node.
        first_boundary = (
            result.prefix_len // C128_GROUP_TOKENS + 1
        ) * C128_GROUP_TOKENS
        for boundary in range(first_boundary, len(params.key) + 1, C128_GROUP_TOKENS):
            boundary_node = self._ensure_boundary_node(node, boundary, cache_actions)
            page_index = boundary // C128_GROUP_TOKENS - 1
            self._attach(boundary_node, params.c128_value[page_index : page_index + 1])

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        # Every stored value belongs to the old child's end boundary. Splitting
        # inside that group leaves the page on the child.
        return

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[self.component_type].append(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= len(
                cd.value
            )
            cd.value = None
        # C128 pages are auxiliary to Full tokens and must not inflate the
        # public token-eviction count.
        return 0, 0

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int:
        logical_len = token_ids_len
        if self.tree_core.is_eagle and logical_len > 0:
            logical_len -= 1
        cache_len = logical_len // C128_GROUP_TOKENS * C128_GROUP_TOKENS
        num_pages = cache_len // C128_GROUP_TOKENS
        insert_params.c128_value = self.cache.req_to_token_pool.req_to_c128_sidecar[
            int(req.req_pool_idx), :num_pages
        ].clone()
        return cache_len + 1 if self.tree_core.is_eagle and cache_len > 0 else cache_len

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            for page_ids in action.indices:
                self.allocator.release_c128_pages(page_ids)
            return
        raise AssertionError(
            f"C128SidecarComponent: unhandled action {type(action).__name__}"
        )

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def _evict_device_start(self, request_cnt):
        pass

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        pass

    def acquire_component_lock(self, node, result, lock_host=False):
        return result

    def release_component_lock(self, node, params, lock_host=False) -> None:
        pass

    def free_host_values(self, host_values) -> None:
        pass
