from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def get_all_node_hit_counts(ctx: "ScriptedContext") -> Dict[int, int]:
    hit_counts: Dict[int, int] = {}
    stack = list(ctx._scheduler.tree_cache.root_node.children.values())
    while stack:
        node = stack.pop()
        hit_counts[node.id] = node.hit_count
        stack.extend(node.children.values())
    return hit_counts


def get_all_node_lock_refs(ctx: "ScriptedContext") -> Dict[int, int]:
    lock_refs: Dict[int, int] = {}
    stack = list(ctx._scheduler.tree_cache.root_node.children.values())
    while stack:
        node = stack.pop()
        lock_refs[node.id] = node.lock_ref
        stack.extend(node.children.values())
    return lock_refs
