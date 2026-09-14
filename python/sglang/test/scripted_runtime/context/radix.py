from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

from sglang.srt.mem_cache.swa_radix_cache import TreeNode as SWATreeNode
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def get_all_node_hit_counts(ctx: ScriptedContext) -> Dict[int, int]:
    return _collect_node_attr(ctx, lambda node: node.hit_count)


def get_all_node_lock_refs(ctx: ScriptedContext) -> Dict[int, int]:
    return _collect_node_attr(ctx, _node_lock_ref)


def _node_lock_ref(node: Any) -> int:
    if isinstance(node, SWATreeNode):
        return node.full_lock_ref + node.swa_lock_ref
    if isinstance(node, UnifiedTreeNode):
        return sum(cd.lock_ref for cd in node.component_data)
    return node.lock_ref


def resolve_node(tree_cache: Any, node_handle: Any) -> Any:
    """Resolve whatever a req or match result carries into a tree node.

    ``UnifiedRadixCache`` hands out NodeIds (ints); every other cache hands out
    the node object itself. Returns None when the handle no longer maps to a
    live node, which only happens once the node has been freed -- and a freed
    node holds no locks.
    """
    resolve = getattr(tree_cache, "resolve_node_handle", None)
    if resolve is None:
        return node_handle
    try:
        return resolve(node_handle)
    except KeyError:
        return None


def to_node_handle(tree_cache: Any, node: Any) -> Any:
    """Inverse of `resolve_node`: what the tree_cache lock APIs accept."""
    if isinstance(node, UnifiedTreeNode):
        return node.id
    return node


def _collect_node_attr(
    ctx: ScriptedContext, get_value: Callable[[Any], int]
) -> Dict[int, int]:
    values: Dict[int, int] = {}
    stack = list(ctx.scheduler.tree_cache.root_node.children.values())
    while stack:
        node = stack.pop()
        values[node.id] = get_value(node)
        stack.extend(node.children.values())
    return values
