"""Free functions for radix-cache inspection and manipulation.

These read radix node state (hit counts, lock refs) and seed / evict
prefix entries. Each takes the facade ``ctx`` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def get_all_node_hit_counts(ctx: "ScriptedContext") -> Dict[int, int]:
    """Map every non-root radix node id to its ``hit_count``.

    The chunked self-referencing guard skips ``_inc_hit_count`` on every
    ``cache_unfinished_req`` insert, so only the single non-chunked
    ``cache_finished_req`` insert bumps hit counts — touching each node on
    the committed path exactly once. On an otherwise-empty tree every node
    therefore sits at ``hit_count == 1``; without the guard the early
    prefix nodes would be re-bumped once per chunk and exceed 1.
    """
    hit_counts: Dict[int, int] = {}
    stack = list(ctx._scheduler.tree_cache.root_node.children.values())
    while stack:
        node = stack.pop()
        hit_counts[node.id] = node.hit_count
        stack.extend(node.children.values())
    return hit_counts


def get_all_node_lock_refs(ctx: "ScriptedContext") -> Dict[int, int]:
    """Map every non-root radix node id to its ``lock_ref``.

    A radix node is locked while a req still holds its KV path and
    released (``lock_ref`` decremented back to 0) once that req
    finishes. With no req in flight every node must therefore sit at
    ``lock_ref == 0``; a stash committed on an un-scheduled chunked
    req would ``inc_lock_ref`` a path without a matching release and
    leave a node locked after the engine drains.
    """
    lock_refs: Dict[int, int] = {}
    stack = list(ctx._scheduler.tree_cache.root_node.children.values())
    while stack:
        node = stack.pop()
        lock_refs[node.id] = node.lock_ref
        stack.extend(node.children.values())
    return lock_refs
