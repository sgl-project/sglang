from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def get_all_node_hit_counts(ctx: ScriptedContext) -> dict[int, int]:
    tree_cache = ctx.scheduler.tree_cache
    core = getattr(tree_cache, "tree_core", None)
    return {
        node if core is not None else node.id: (
            core.get_node_hit_count(node) if core is not None else node.hit_count
        )
        for node in iter_node_handles(tree_cache)
    }


def get_all_node_lock_refs(ctx: ScriptedContext) -> dict[int, int]:
    tree_cache = ctx.scheduler.tree_cache
    core = getattr(tree_cache, "tree_core", None)
    return {
        node if core is not None else node.id: get_node_lock_ref(tree_cache, node)
        for node in iter_node_handles(tree_cache)
    }


def _node_lock_ref(node: Any) -> int:
    if isinstance(node, UnifiedTreeNode):
        return sum(cd.lock_ref for cd in node.component_data)
    return node.lock_ref


def get_node_lock_ref(tree_cache: Any, node_handle: Any) -> int:
    """Read locks through the cache's native handle, including stale handles."""
    if node_handle is None:
        return 0
    core = getattr(tree_cache, "tree_core", None)
    if core is not None:
        if not core.contains_node(node_handle):
            return 0
        return sum(
            core.get_component_device_lock_ref(node_handle, component)
            for component in tree_cache.tree_components
        )
    return _node_lock_ref(node_handle)


def iter_node_handles(tree_cache: Any) -> Iterator[Any]:
    """Walk non-root nodes as IDs for unified caches or native legacy nodes."""
    core = getattr(tree_cache, "tree_core", None)
    stack = (
        list(core.get_child_node_ids(core.root_node_handle()))
        if core is not None
        else list(tree_cache.root_node.children.values())
    )
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            core.get_child_node_ids(node)
            if core is not None
            else node.children.values()
        )
