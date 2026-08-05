"""KVFlow workflow-aware eviction manager for multi-agent LLM workloads.

Implements the eviction half of KVFlow (arXiv:2507.07400): maintains a
per-agent set of leaf nodes and updates ``node.kvflow_priority`` on the
prefix path whenever the external scheduler posts new ``steps_to_execution``
values via ``/v1/kvflow/update``.

Priority scheme (higher = evicted later):
  0  — no KVFlow protection (default, dynamic suffix, or far-future)
  1  — far future (steps >= hold_step)
  2  — steps = hold_step - 2
  ...
  hold_step — steps = 1 (next agent to run, maximum protection)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


def _steps_to_priority(steps: int, hold_step: int) -> int:
    """Convert steps_to_execution to a kvflow_priority value.

    Steps closer to 0 (sooner to execute) get higher priority (protected longer).
    Steps at or beyond ``hold_step`` get priority 1 (low, but not unprotected).
    Steps=0 means currently running — caller skips those.
    """
    if steps >= hold_step:
        return 1
    return hold_step - steps + 1


class KVFlowAgentManager:
    """Tracks per-agent leaf nodes and refreshes their KVFlow eviction priorities.

    Thread-safe: ``update`` and ``register_leaf`` may be called from different
    threads (the scheduler event loop vs. the HTTP worker that receives /update).
    """

    def __init__(self, hold_step: int = 4) -> None:
        self.hold_step = hold_step
        self._agent_to_leaves: dict[str, set[UnifiedTreeNode]] = defaultdict(set)
        self._lock = threading.Lock()

    def register_leaf(self, agent_id: str, leaf: Optional[UnifiedTreeNode]) -> None:
        """Record the deepest cached node for an agent after a request completes."""
        if leaf is None:
            return
        with self._lock:
            self._agent_to_leaves[agent_id].add(leaf)

    def update(self, agent_updates: dict[str, int]) -> None:
        """Apply a full snapshot of {agent_id: steps_to_execution} from the workflow scheduler.

        Resets all previously tagged nodes to kvflow_priority=0, then re-applies
        priorities for agents present in agent_updates with steps > 0.
        """
        with self._lock:
            self._reset_all_node_priorities()
            self._remove_stale_agents(set(agent_updates))
            self._apply_fresh_priorities(agent_updates)

    def _reset_all_node_priorities(self) -> None:
        for leaves in self._agent_to_leaves.values():
            for leaf in leaves:
                self._walk_path(leaf, action=_clear_priority)

    def _remove_stale_agents(self, active_agents: set[str]) -> None:
        stale = [aid for aid in self._agent_to_leaves if aid not in active_agents]
        for aid in stale:
            del self._agent_to_leaves[aid]

    def _apply_fresh_priorities(self, agent_updates: dict[str, int]) -> None:
        for agent_id, steps in agent_updates.items():
            if steps <= 0:
                continue
            priority = _steps_to_priority(steps, self.hold_step)
            for leaf in self._agent_to_leaves.get(agent_id, ()):
                self._walk_path(leaf, action=_raise_priority_to(priority))

    @staticmethod
    def _walk_path(
        leaf: UnifiedTreeNode, action
    ) -> None:
        """Walk from leaf to root (exclusive), applying action to each node."""
        cur = leaf
        while cur is not None and cur.parent is not None:
            action(cur)
            cur = cur.parent


def _clear_priority(node: UnifiedTreeNode) -> None:
    node.kvflow_priority = 0


def _raise_priority_to(priority: int):
    def _apply(node: UnifiedTreeNode) -> None:
        if node.kvflow_priority < priority:
            node.kvflow_priority = priority

    return _apply
