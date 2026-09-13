from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from sglang.test.scripted_runtime.context.radix import _node_lock_ref, to_node_handle

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class ScriptedLockRefExhauster:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        # (node, dec receipt) pairs; the receipt bounds the release walk.
        self._locked: List[tuple[Any, Any]] = []

    def exhaust(self, *, leave_refs: int) -> None:
        tree_cache = self.scheduler.tree_cache
        if tree_cache.disable:
            return

        while True:
            evictable = self._evictable_nodes()
            if len(evictable) <= leave_refs:
                return

            target = evictable[0]
            result = tree_cache.inc_lock_ref(to_node_handle(tree_cache, target))

            newly_locked = [node for node in evictable if _node_lock_ref(node) > 0]
            if not newly_locked:
                return
            self._locked.append((target, result.to_dec_params()))

    def release(self) -> None:
        tree_cache = self.scheduler.tree_cache
        for node, dec_params in self._locked:
            tree_cache.dec_lock_ref(to_node_handle(tree_cache, node), dec_params)
        self._locked.clear()

    def _evictable_nodes(self) -> List[Any]:
        evictable: List[Any] = []
        stack = list(self.scheduler.tree_cache.root_node.children.values())
        while stack:
            node = stack.pop()
            if _node_lock_ref(node) == 0:
                evictable.append(node)
            stack.extend(node.children.values())
        return evictable
