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
