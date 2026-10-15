from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.test.scripted_runtime.context.radix import (
    get_node_lock_ref,
    iter_node_handles,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class ScriptedLockRefExhauster:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        # (node, dec receipt) pairs; the receipt bounds the release walk.
        self._locked: list[tuple[Any, Any]] = []

    def exhaust(self, leave_refs: int) -> None:
        tree_cache = self.scheduler.tree_cache
        if tree_cache.disable:
            return

        while True:
            evictable = self._evictable_nodes()
            if len(evictable) <= leave_refs:
                return

            target = evictable[0]
            result = tree_cache.inc_lock_ref(target)

            newly_locked = [
                node for node in evictable if get_node_lock_ref(tree_cache, node) > 0
            ]
            if not newly_locked:
                return
            self._locked.append((target, result.to_dec_params()))

    def release(self) -> None:
        tree_cache = self.scheduler.tree_cache
        for node, dec_params in self._locked:
            tree_cache.dec_lock_ref(node, dec_params)
        self._locked.clear()

    def _evictable_nodes(self) -> list[Any]:
        tree_cache = self.scheduler.tree_cache
        return [
            node
            for node in iter_node_handles(tree_cache)
            if get_node_lock_ref(tree_cache, node) == 0
        ]
