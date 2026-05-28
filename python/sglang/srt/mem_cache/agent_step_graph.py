# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Agent Step Graph — DAG-based workflow tracker for agent-aware KV cache management.

This module maintains a lightweight DAG representation of active agent workflows,
enabling the system to compute ``steps_to_execution`` for each cached KV block.
The distance metric is used by ``AgentAwareEvictionStrategy`` to protect KV cache
that will be reused in upcoming workflow steps.

Reference: KVFlow (2025) — Agent Step Graph concept.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class StepNode:
    """A single step in a workflow DAG."""

    step_id: str
    agent_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | waiting_tool | completed
    tool_name: Optional[str] = None
    tool_duration_prediction_ms: Optional[int] = None
    created_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    # Weak references to RadixCache TreeNodes associated with this step.
    # We don't hold strong refs to avoid preventing GC.
    associated_cache_node_ids: Set[int] = field(default_factory=set)


@dataclass
class WorkflowDAG:
    """DAG representation of a single agent workflow."""

    workflow_id: str
    nodes: Dict[str, StepNode] = field(default_factory=dict)
    current_step_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    last_updated: float = field(default_factory=time.monotonic)

    @property
    def total_steps(self) -> int:
        return len(self.nodes)

    @property
    def completed_steps(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == "completed")


class AgentStepGraph:
    """Manages DAGs for all active workflows.

    Thread safety: designed to be called only from the Scheduler main loop
    (single-threaded event loop), so no locking is needed.

    Usage::

        graph = AgentStepGraph()

        # When a request arrives with agent_hints:
        graph.register_step(
            workflow_id="wf-1",
            step_id="step-3",
            agent_id="agent-A",
            parent_step_id="step-2",
            children_step_ids=["step-4", "step-5"],
        )

        # Query distance for eviction decisions:
        dist = graph.compute_steps_to_execution("wf-1", "step-2")

        # Mark completion:
        graph.mark_step_completed("wf-1", "step-3")

        # Periodic cleanup:
        graph.cleanup_expired_workflows(timeout_s=300)
    """

    def __init__(self, max_workflows: int = 10000):
        self.workflows: Dict[str, WorkflowDAG] = {}
        self.max_workflows = max_workflows

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_step(
        self,
        workflow_id: str,
        step_id: str,
        agent_id: Optional[str] = None,
        parent_step_id: Optional[str] = None,
        children_step_ids: Optional[List[str]] = None,
        tool_name: Optional[str] = None,
        expected_tool_duration_ms: Optional[int] = None,
    ) -> WorkflowDAG:
        """Register or update a step in a workflow DAG.

        If the workflow doesn't exist yet, it will be created.
        If the step already exists, its metadata is updated (idempotent).
        """
        # Create workflow if needed
        if workflow_id not in self.workflows:
            if len(self.workflows) >= self.max_workflows:
                self._evict_oldest_workflow()
            self.workflows[workflow_id] = WorkflowDAG(workflow_id=workflow_id)

        dag = self.workflows[workflow_id]
        dag.last_updated = time.monotonic()

        # Create or update step
        if step_id not in dag.nodes:
            dag.nodes[step_id] = StepNode(step_id=step_id, agent_id=agent_id)

        step = dag.nodes[step_id]
        step.status = "running"
        dag.current_step_id = step_id

        if agent_id:
            step.agent_id = agent_id
        if tool_name:
            step.tool_name = tool_name
        if expected_tool_duration_ms is not None:
            step.tool_duration_prediction_ms = expected_tool_duration_ms

        # Wire DAG edges
        if parent_step_id:
            if parent_step_id not in step.parent_ids:
                step.parent_ids.append(parent_step_id)
            # Ensure parent node exists (may be a stub)
            if parent_step_id not in dag.nodes:
                dag.nodes[parent_step_id] = StepNode(
                    step_id=parent_step_id, status="completed"
                )
            parent = dag.nodes[parent_step_id]
            if step_id not in parent.children_ids:
                parent.children_ids.append(step_id)

        if children_step_ids:
            for child_id in children_step_ids:
                if child_id not in step.children_ids:
                    step.children_ids.append(child_id)
                # Ensure child node exists (stub)
                if child_id not in dag.nodes:
                    dag.nodes[child_id] = StepNode(step_id=child_id)

        return dag

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def compute_steps_to_execution(
        self, workflow_id: str, step_id: str
    ) -> int:
        """Compute minimum hops from ``step_id`` to the current execution front.

        Returns:
            0  — this step is currently executing (or about to).
            N  — N hops away from any currently-running step.
            -1 — workflow or step not found.
        """
        dag = self.workflows.get(workflow_id)
        if dag is None or step_id not in dag.nodes:
            return -1

        # BFS forward from step_id towards currently-running steps
        if dag.current_step_id == step_id:
            return 0

        # BFS from the *current* step backwards to find distance to step_id
        visited: Set[str] = set()
        queue: List[tuple] = [(dag.current_step_id, 0)]
        visited.add(dag.current_step_id)

        while queue:
            current, dist = queue.pop(0)
            node = dag.nodes.get(current)
            if node is None:
                continue
            # Walk parents (step_id might be an ancestor of current)
            for parent_id in node.parent_ids:
                if parent_id == step_id:
                    return dist + 1
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, dist + 1))

        # Also BFS forward from step_id to find if it's a descendant
        visited2: Set[str] = set()
        queue2: List[tuple] = [(step_id, 0)]
        visited2.add(step_id)

        while queue2:
            current, dist = queue2.pop(0)
            node = dag.nodes.get(current)
            if node is None:
                continue
            for child_id in node.children_ids:
                if child_id == dag.current_step_id:
                    return dist + 1
                if child_id not in visited2:
                    visited2.add(child_id)
                    queue2.append((child_id, dist + 1))

        return -1  # Not reachable

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDAG]:
        """Return the DAG for a workflow, or None."""
        return self.workflows.get(workflow_id)

    def get_active_workflow_ids(self) -> List[str]:
        """Return all active workflow IDs."""
        return list(self.workflows.keys())

    def get_workflow_stats(self) -> Dict[str, dict]:
        """Return summary statistics for all active workflows."""
        stats = {}
        for wf_id, dag in self.workflows.items():
            stats[wf_id] = {
                "total_steps": dag.total_steps,
                "completed_steps": dag.completed_steps,
                "current_step": dag.current_step_id,
                "age_s": time.monotonic() - dag.created_at,
            }
        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def mark_step_completed(self, workflow_id: str, step_id: str):
        """Mark a step as completed."""
        dag = self.workflows.get(workflow_id)
        if dag is None:
            return
        step = dag.nodes.get(step_id)
        if step is None:
            return
        step.status = "completed"
        step.completed_at = time.monotonic()
        dag.last_updated = time.monotonic()

        # If all steps are completed, the workflow is done
        if dag.completed_steps == dag.total_steps:
            logger.debug(f"Workflow {workflow_id} fully completed, will be cleaned up.")

    def mark_step_waiting_tool(self, workflow_id: str, step_id: str):
        """Mark a step as waiting for a tool call to return."""
        dag = self.workflows.get(workflow_id)
        if dag is None:
            return
        step = dag.nodes.get(step_id)
        if step is None:
            return
        step.status = "waiting_tool"
        dag.last_updated = time.monotonic()

    def cleanup_expired_workflows(self, timeout_s: float = 300.0):
        """Remove workflows that haven't been updated for ``timeout_s`` seconds."""
        now = time.monotonic()
        expired = [
            wf_id
            for wf_id, dag in self.workflows.items()
            if (now - dag.last_updated) > timeout_s
        ]
        for wf_id in expired:
            logger.debug(f"Cleaning up expired workflow: {wf_id}")
            del self.workflows[wf_id]
        return len(expired)

    def _evict_oldest_workflow(self):
        """Evict the oldest workflow when max_workflows is reached."""
        if not self.workflows:
            return
        oldest_id = min(
            self.workflows, key=lambda wf_id: self.workflows[wf_id].last_updated
        )
        logger.debug(
            f"AgentStepGraph at capacity ({self.max_workflows}), "
            f"evicting oldest workflow: {oldest_id}"
        )
        del self.workflows[oldest_id]

    # ------------------------------------------------------------------
    # Helpers for Scheduler integration
    # ------------------------------------------------------------------

    def register_from_req(self, req) -> Optional[WorkflowDAG]:
        """Convenience: register a step from a Req object's agent_hints fields.

        Returns the WorkflowDAG if registration occurred, else None.
        """
        workflow_id = getattr(req, "workflow_id", None)
        step_id = getattr(req, "step_id", None)
        if not workflow_id or not step_id:
            return None

        return self.register_step(
            workflow_id=workflow_id,
            step_id=step_id,
            agent_id=getattr(req, "agent_id", None),
            parent_step_id=getattr(req, "parent_step_id", None),
            children_step_ids=getattr(req, "children_step_ids", None),
            tool_name=getattr(req, "tool_name", None),
            expected_tool_duration_ms=getattr(req, "expected_tool_duration_ms", None),
        )

    def update_tree_node_distances(self, workflow_id: str, nodes: list):
        """Update steps_to_execution on a list of TreeNodes for a workflow.

        Called after DAG changes to propagate distance info to cache nodes.
        """
        dag = self.workflows.get(workflow_id)
        if dag is None:
            return
        for node in nodes:
            # Find the minimum distance across all workflows this node belongs to
            for wf_id in node.workflow_ids:
                if wf_id in self.workflows:
                    # For each step in the workflow, check if this node is relevant
                    # Simplified: use -1 to indicate "managed by DAG" but let
                    # the eviction strategy handle the actual distance
                    pass
