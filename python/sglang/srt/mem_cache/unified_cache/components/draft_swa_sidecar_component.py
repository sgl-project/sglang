from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransferResult
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class DraftSWASidecarComponent:
    """Coverage tracker for draft SWA that depends on target SWA slots."""

    _DEVICE_KEY = "draft_swa_device_covered"
    _HOST_KEY = "draft_swa_host_covered"

    def __init__(self, cache: UnifiedRadixCache):
        self.cache = cache

    def _metadata(self, node: UnifiedTreeNode) -> dict:
        return node.component_data[ComponentType.SWA].metadata

    @property
    def window_span(self) -> int:
        page_size = self.cache.page_size
        return (
            self.cache.sliding_window_size + page_size - 1
        ) // page_size * page_size

    def mark_device(self, node: UnifiedTreeNode) -> None:
        self._metadata(node)[self._DEVICE_KEY] = True

    def mark_host(self, node: UnifiedTreeNode) -> None:
        self._metadata(node)[self._HOST_KEY] = True

    def clear_device(self, node: UnifiedTreeNode) -> None:
        self._metadata(node).pop(self._DEVICE_KEY, None)

    def clear_host(self, node: UnifiedTreeNode) -> None:
        self._metadata(node).pop(self._HOST_KEY, None)

    def has_host_window(self, node: UnifiedTreeNode) -> bool:
        return self._window_covered(node, host=True)

    def has_device_window(self, node: UnifiedTreeNode) -> bool:
        return self._window_covered(node, host=False)

    def has_loadable_window(self, node: UnifiedTreeNode) -> bool:
        covered = 0
        root = self.cache.tree_core.root_node
        cur = node
        while cur is not root and covered < self.window_span:
            cd = cur.component_data[ComponentType.SWA]
            if cd.value is not None:
                if not cd.metadata.get(self._DEVICE_KEY, False):
                    return False
                covered += len(cd.value)
            elif cd.host_value is not None:
                if not cd.metadata.get(self._HOST_KEY, False):
                    return False
                covered += len(cd.host_value)
            else:
                return False
            cur = cur.parent
        return covered >= self.window_span

    def _window_covered(self, node: UnifiedTreeNode, *, host: bool) -> bool:
        key = self._HOST_KEY if host else self._DEVICE_KEY
        covered = 0
        target = self.window_span
        root = self.cache.tree_core.root_node
        cur = node
        while cur is not root and covered < target:
            cd = cur.component_data[ComponentType.SWA]
            value = cd.host_value if host else cd.value
            if value is None or not cd.metadata.get(key, False):
                return False
            covered += len(value)
            cur = cur.parent
        return covered >= target

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        child_meta = self._metadata(child)
        parent_meta = self._metadata(new_parent)
        for key in (self._DEVICE_KEY, self._HOST_KEY):
            if child_meta.get(key, False):
                parent_meta[key] = True

    def mark_host_backup(self, node: UnifiedTreeNode, transferred: bool) -> None:
        if transferred:
            self.mark_host(node)

    def mark_device_load(self, node: UnifiedTreeNode, transferred: bool) -> None:
        if not transferred:
            return
        root = self.cache.tree_core.root_node
        covered = 0
        target = self.window_span
        cur = node
        while cur is not root and covered < target:
            cd = cur.component_data[ComponentType.SWA]
            if cd.value is not None:
                self.mark_device(cur)
                covered += len(cd.value)
            cur = cur.parent

    def commit_prefetch(
        self,
        node: UnifiedTreeNode,
        transfer,
        result: PoolTransferResult,
    ) -> None:
        if transfer is None or transfer.keys is None:
            return
        hit_pages = result.extra_pool_hit_pages.get(PoolName.DRAFT_SWA, 0)
        if hit_pages < len(transfer.keys):
            return
        root = self.cache.tree_core.root_node
        covered = 0
        target = self.window_span
        cur = node
        while cur is not root and covered < target:
            cd = cur.component_data[ComponentType.SWA]
            if cd.host_value is not None:
                self.mark_host(cur)
                covered += len(cd.host_value)
            cur = cur.parent
