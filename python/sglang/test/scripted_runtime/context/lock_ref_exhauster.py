from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from sglang.test.scripted_runtime.context.radix import _node_lock_ref

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class ScriptedLockRefExhauster:
    """Pins evictable radix nodes via the cache's real ``inc_lock_ref`` to create
    deterministic eviction pressure, and releases them so the next script starts
    from a fully-evictable tree.

    This is distinct from ``exhaust_kv``: the KV exhauster makes free pages
    *disappear* by grabbing them from the allocator, whereas this exhauster
    leaves the KV present but *pinned* (protected, non-evictable). It exercises
    the path where the cache is full of locked prefixes and the engine must make
    progress without being able to evict them.

    Uses the cache's public ``inc_lock_ref`` / ``dec_lock_ref`` so the held
    nodes move between the evictable and protected accounting exactly like
    engine-owned locks; this is what makes "leave exactly N nodes evictable"
    precise and reversible.
    """

    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = scheduler
        self._locked: List[Any] = []

    def exhaust(self, *, leave_refs: int) -> None:
        # "leave_refs" = the number of radix nodes left unlocked (evictable)
        # after exhaustion. We repeatedly pick a still-evictable node and lock it
        # via inc_lock_ref until at most `leave_refs` evictable nodes remain.
        #
        # inc_lock_ref(node) walks node -> root, locking every ancestor too, so a
        # single call can lock several nodes at once. We therefore re-scan the
        # evictable set after each lock instead of assuming a fixed count, and
        # only record nodes that this call actually transitioned from evictable
        # to locked, so release() can undo exactly what we did.
        tree_cache = self.scheduler.tree_cache
        # A disabled cache (ChunkCache, or radix disabled) has no evictable
        # prefix tree to pin, and its inc_lock_ref is a no-op. Nothing to do.
        if tree_cache.disable:
            return

        while True:
            evictable = self._evictable_nodes()
            if len(evictable) <= leave_refs:
                return

            # Any evictable node reduces the evictable count by at least one
            # (itself), so pick the first and re-scan.
            target = evictable[0]
            tree_cache.inc_lock_ref(target)

            newly_locked = [node for node in evictable if _node_lock_ref(node) > 0]
            if not newly_locked:
                # Defensive: inc_lock_ref made no node non-evictable (should not
                # happen for a real evictable node). Stop to avoid an infinite
                # loop rather than hand-mutating state.
                return
            self._locked.append(target)

    def release(self) -> None:
        # Idempotent cleanup run before each script resets the engine. Undo each
        # inc_lock_ref with a matching dec_lock_ref on the same node, then clear
        # the record so a second call is a no-op.
        tree_cache = self.scheduler.tree_cache
        for node in self._locked:
            tree_cache.dec_lock_ref(node)
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
